"""
Unscented Kalman Filter (UKF) for Non-Linear Object Tracking
=============================================================
Implements UKF with:
- Non-linear motion model with acceleration
- Adaptive process noise based on motion patterns
- Extended state for aspect ratio preservation
- Robust to rapid direction changes (essential for hockey)

State Vector: [x, y, s, r, vx, vy, vs]
- (x, y): center position
- s: scale (area = w*h)
- r: aspect ratio (w/h) - treated as constant
- (vx, vy): velocity
- vs: scale velocity

This parameterization is more stable than [x,y,w,h,vx,vy,vw,vh]
because aspect ratio changes slowly for rigid objects.
"""

import numpy as np
from typing import Tuple, Optional
import scipy.linalg


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter for bounding box tracking.
    
    Uses sigma point propagation for better non-linear estimation.
    More accurate than EKF for rapid motion changes in hockey.
    """
    
    def __init__(
        self,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        """
        Initialize UKF with tuning parameters.
        
        Args:
            alpha: Spread of sigma points (small = points closer to mean)
            beta: Prior knowledge of distribution (2 optimal for Gaussian)
            kappa: Secondary scaling parameter (usually 0)
        """
        self.ndim_state = 7  # [x, y, s, r, vx, vy, vs]
        self.ndim_meas = 4   # [x, y, s, r]
        
        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # Compute weights
        self._compute_weights()
        
        # Motion uncertainty parameters (adaptive)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
        self._std_weight_scale = 1.0 / 20
        
        # Measurement noise
        self._std_weight_meas = 1.0 / 20
        
        # Acceleration estimation for adaptive noise
        self._prev_velocity = None
        self._acceleration_estimate = 0.0
    
    def _compute_weights(self):
        """Compute sigma point weights."""
        n = self.ndim_state
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        
        # Mean weights
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lambda_)))
        self.Wm[0] = lambda_ / (n + lambda_)
        
        # Covariance weights
        self.Wc = np.full(2 * n + 1, 1.0 / (2 * (n + lambda_)))
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        
        self.lambda_ = lambda_
    
    def _generate_sigma_points(
        self, 
        mean: np.ndarray, 
        covariance: np.ndarray
    ) -> np.ndarray:
        """
        Generate sigma points around mean.
        
        Args:
            mean: State mean (7,)
            covariance: State covariance (7, 7)
            
        Returns:
            Sigma points (15, 7) - 2n+1 points
        """
        n = self.ndim_state
        lambda_ = self.lambda_
        
        # Ensure covariance is positive definite
        try:
            sqrt_cov = scipy.linalg.cholesky(
                (n + lambda_) * covariance, lower=True
            )
        except np.linalg.LinAlgError:
            # Add small regularization if not positive definite
            covariance = covariance + np.eye(n) * 1e-6
            sqrt_cov = scipy.linalg.cholesky(
                (n + lambda_) * covariance, lower=True
            )
        
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = mean
        
        for i in range(n):
            sigma_points[i + 1] = mean + sqrt_cov[:, i]
            sigma_points[n + i + 1] = mean - sqrt_cov[:, i]
        
        return sigma_points
    
    def _motion_model(self, state: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Non-linear motion model with constant velocity.
        
        Args:
            state: [x, y, s, r, vx, vy, vs]
            dt: Time step
            
        Returns:
            Predicted state
        """
        x, y, s, r, vx, vy, vs = state
        
        # Constant velocity prediction
        x_new = x + vx * dt
        y_new = y + vy * dt
        s_new = s + vs * dt
        
        # Scale must be positive
        s_new = max(s_new, 1.0)
        
        # Aspect ratio is constant (rigid body assumption)
        r_new = r
        
        return np.array([x_new, y_new, s_new, r_new, vx, vy, vs])
    
    def _measurement_model(self, state: np.ndarray) -> np.ndarray:
        """
        Measurement model: observe [x, y, s, r] from state.
        
        Args:
            state: Full state vector
            
        Returns:
            Measurement vector [x, y, s, r]
        """
        return state[:4]
    
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create track from initial bounding box.
        
        Args:
            measurement: [x, y, w, h] bounding box (center format)
            
        Returns:
            (mean, covariance): Initial state distribution
        """
        x, y, w, h = measurement
        
        # Convert to [x, y, s, r] parameterization
        s = w * h  # Scale (area)
        r = w / max(h, 1e-6)  # Aspect ratio
        
        # Initial state: [x, y, s, r, vx, vy, vs]
        mean = np.array([x, y, s, r, 0., 0., 0.])
        
        # Initial covariance (larger for velocities since unknown)
        std = [
            2 * self._std_weight_position * w,      # x
            2 * self._std_weight_position * h,      # y
            2 * self._std_weight_scale * s,         # s
            0.1,                                     # r (aspect ratio quite certain)
            10 * self._std_weight_velocity * w,     # vx
            10 * self._std_weight_velocity * h,     # vy
            10 * self._std_weight_velocity * s,     # vs
        ]
        covariance = np.diag(np.square(std))
        
        return mean, covariance
    
    def predict(
        self, 
        mean: np.ndarray, 
        covariance: np.ndarray,
        dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        UKF prediction step using sigma point propagation.
        
        Args:
            mean: Current state mean (7,)
            covariance: Current state covariance (7, 7)
            dt: Time step
            
        Returns:
            (predicted_mean, predicted_covariance)
        """
        # Generate sigma points
        sigma_points = self._generate_sigma_points(mean, covariance)
        
        # Propagate sigma points through motion model
        propagated = np.array([
            self._motion_model(sp, dt) for sp in sigma_points
        ])
        
        # Weighted mean
        predicted_mean = np.sum(self.Wm[:, np.newaxis] * propagated, axis=0)
        
        # Weighted covariance
        diff = propagated - predicted_mean
        predicted_cov = np.zeros((self.ndim_state, self.ndim_state))
        for i, w in enumerate(self.Wc):
            predicted_cov += w * np.outer(diff[i], diff[i])
        
        # Add process noise (adaptive based on motion)
        process_noise = self._compute_process_noise(mean, dt)
        predicted_cov += process_noise
        
        return predicted_mean, predicted_cov
    
    def _compute_process_noise(
        self, 
        state: np.ndarray, 
        dt: float
    ) -> np.ndarray:
        """
        Compute adaptive process noise based on motion.
        
        Higher noise for faster moving objects (more uncertainty).
        """
        x, y, s, r, vx, vy, vs = state
        
        # Velocity magnitude affects uncertainty
        velocity = np.sqrt(vx**2 + vy**2)
        velocity_factor = 1.0 + 0.1 * velocity / 100.0  # Scale by typical velocity
        
        # Convert scale to approximate width/height for noise computation
        w = np.sqrt(s * r)
        h = np.sqrt(s / max(r, 1e-6))
        
        std = [
            self._std_weight_position * w * velocity_factor,    # x
            self._std_weight_position * h * velocity_factor,    # y
            self._std_weight_scale * s * velocity_factor,       # s
            0.01,                                                # r (very stable)
            self._std_weight_velocity * w * velocity_factor,    # vx
            self._std_weight_velocity * h * velocity_factor,    # vy
            self._std_weight_velocity * s,                       # vs
        ]
        
        return np.diag(np.square(std)) * dt
    
    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        UKF update step with measurement.
        
        Args:
            mean: Predicted state mean (7,)
            covariance: Predicted state covariance (7, 7)
            measurement: [x, y, w, h] observed bounding box
            
        Returns:
            (updated_mean, updated_covariance)
        """
        x, y, w, h = measurement
        
        # Convert to [x, y, s, r] measurement
        s = w * h
        r = w / max(h, 1e-6)
        meas = np.array([x, y, s, r])
        
        # Generate sigma points
        sigma_points = self._generate_sigma_points(mean, covariance)
        
        # Transform to measurement space
        meas_sigma = np.array([
            self._measurement_model(sp) for sp in sigma_points
        ])
        
        # Predicted measurement
        predicted_meas = np.sum(self.Wm[:, np.newaxis] * meas_sigma, axis=0)
        
        # Measurement covariance
        meas_diff = meas_sigma - predicted_meas
        S = np.zeros((self.ndim_meas, self.ndim_meas))
        for i, w in enumerate(self.Wc):
            S += w * np.outer(meas_diff[i], meas_diff[i])
        
        # Add measurement noise
        meas_noise = self._compute_measurement_noise(measurement)
        S += meas_noise
        
        # Cross covariance
        state_diff = sigma_points - mean
        Pxz = np.zeros((self.ndim_state, self.ndim_meas))
        for i, w in enumerate(self.Wc):
            Pxz += w * np.outer(state_diff[i], meas_diff[i])
        
        # Kalman gain
        try:
            K = scipy.linalg.solve(S.T, Pxz.T, assume_a='pos').T
        except:
            K = np.dot(Pxz, np.linalg.pinv(S))
        
        # Innovation
        innovation = meas - predicted_meas
        
        # Update
        new_mean = mean + np.dot(K, innovation)
        new_cov = covariance - np.linalg.multi_dot([K, S, K.T])
        
        # Ensure scale and aspect ratio remain positive
        new_mean[2] = max(new_mean[2], 1.0)  # s > 0
        new_mean[3] = max(new_mean[3], 0.1)  # r > 0
        
        return new_mean, new_cov
    
    def _compute_measurement_noise(self, measurement: np.ndarray) -> np.ndarray:
        """Compute measurement noise covariance."""
        x, y, w, h = measurement
        s = w * h
        
        std = [
            self._std_weight_meas * w,   # x
            self._std_weight_meas * h,   # y
            self._std_weight_meas * s,   # s
            0.1,                          # r
        ]
        return np.diag(np.square(std))
    
    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False
    ) -> np.ndarray:
        """
        Compute Mahalanobis distance for gating.
        
        Args:
            mean: State mean (7,)
            covariance: State covariance (7, 7)
            measurements: [N, 4] array of [x, y, w, h]
            only_position: Only use position for gating
            
        Returns:
            Squared Mahalanobis distances (N,)
        """
        # Convert measurements to [x, y, s, r]
        meas_converted = np.zeros((len(measurements), 4))
        for i, m in enumerate(measurements):
            x, y, w, h = m
            meas_converted[i] = [x, y, w*h, w/max(h, 1e-6)]
        
        # Project state to measurement space
        predicted_meas = mean[:4]
        
        # Measurement covariance (simplified projection)
        H = np.eye(4, 7)  # Simple observation matrix
        S = np.dot(H, np.dot(covariance, H.T))
        S += self._compute_measurement_noise(measurements[0] if len(measurements) > 0 else np.array([0,0,10,10]))
        
        if only_position:
            predicted_meas = predicted_meas[:2]
            S = S[:2, :2]
            meas_converted = meas_converted[:, :2]
        
        # Mahalanobis distance
        try:
            chol = scipy.linalg.cholesky(S, lower=True)
            diff = meas_converted - predicted_meas
            z = scipy.linalg.solve_triangular(chol, diff.T, lower=True)
            return np.sum(z * z, axis=0)
        except:
            # Fallback to pseudo-inverse
            diff = meas_converted - predicted_meas
            S_inv = np.linalg.pinv(S)
            return np.array([np.dot(d, np.dot(S_inv, d)) for d in diff])
    
    def state_to_bbox(self, state: np.ndarray) -> np.ndarray:
        """
        Convert state [x, y, s, r, ...] to bbox [x, y, w, h].
        
        Args:
            state: State vector
            
        Returns:
            [x, y, w, h] center format
        """
        x, y, s, r = state[:4]
        s = max(s, 1.0)
        r = max(r, 0.1)
        
        w = np.sqrt(s * r)
        h = np.sqrt(s / r)
        
        return np.array([x, y, w, h])
    
    def bbox_to_measurement(self, bbox: np.ndarray) -> np.ndarray:
        """
        Convert bbox [x, y, w, h] to measurement [x, y, s, r].
        
        Args:
            bbox: [x, y, w, h] center format
            
        Returns:
            [x, y, s, r] measurement
        """
        x, y, w, h = bbox
        s = w * h
        r = w / max(h, 1e-6)
        return np.array([x, y, s, r])


class AdaptiveUKF(UnscentedKalmanFilter):
    """
    UKF with adaptive noise estimation.
    
    Automatically adjusts process noise based on innovation sequence.
    Better for handling varying motion patterns in hockey.
    """
    
    def __init__(self, adaptation_rate: float = 0.1, **kwargs):
        """
        Args:
            adaptation_rate: How fast to adapt noise (0-1)
        """
        super().__init__(**kwargs)
        self.adaptation_rate = adaptation_rate
        self.innovation_history = []
        self.max_history = 10
    
    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update with adaptive noise estimation."""
        
        # Compute predicted measurement for innovation
        predicted_meas = mean[:4]
        x, y, w, h = measurement
        actual_meas = np.array([x, y, w*h, w/max(h, 1e-6)])
        
        # Track innovation
        innovation = actual_meas - predicted_meas
        self.innovation_history.append(innovation)
        if len(self.innovation_history) > self.max_history:
            self.innovation_history.pop(0)
        
        # Adapt noise based on innovation magnitude
        if len(self.innovation_history) >= 3:
            innovations = np.array(self.innovation_history)
            innovation_cov = np.cov(innovations.T)
            
            # Increase process noise if innovations are larger than expected
            expected_var = np.diag(covariance[:4, :4])
            actual_var = np.diag(innovation_cov) if innovation_cov.ndim > 1 else innovation_cov
            
            ratio = np.mean(actual_var / (expected_var + 1e-6))
            if ratio > 1.5:
                self._std_weight_position *= (1 + self.adaptation_rate)
                self._std_weight_velocity *= (1 + self.adaptation_rate)
            elif ratio < 0.5:
                self._std_weight_position *= (1 - self.adaptation_rate * 0.5)
                self._std_weight_velocity *= (1 - self.adaptation_rate * 0.5)
        
        return super().update(mean, covariance, measurement)
