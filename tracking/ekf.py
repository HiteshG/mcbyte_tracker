"""
Extended Kalman Filter (EKF) for Non-Linear Hockey Tracking
============================================================
Implements EKF with:
- Non-linear coordinated turn motion model (handles rapid direction changes)
- Adaptive noise based on motion patterns
- State: [x, y, w, h, vx, vy, ax, ay, omega] 
  - Position, size, velocity, acceleration, turn rate
- Handles hockey-specific motion patterns

Why EKF over standard KF for hockey:
1. Players accelerate/decelerate rapidly
2. Sudden direction changes during skating
3. Non-constant velocity assumption fails
"""

import numpy as np
from typing import Tuple, Optional
import scipy.linalg
from dataclasses import dataclass


@dataclass
class EKFConfig:
    """EKF configuration parameters."""
    # State dimensions
    ndim_state: int = 9     # [x, y, w, h, vx, vy, ax, ay, omega]
    ndim_meas: int = 4      # [x, y, w, h]
    
    # Process noise parameters
    std_pos: float = 1.0
    std_vel: float = 1.0  
    std_acc: float = 0.5
    std_omega: float = 0.1  # Turn rate
    std_size: float = 0.05
    
    # Measurement noise
    std_meas_pos: float = 1.0
    std_meas_size: float = 0.1
    
    # Adaptive noise
    adapt_noise: bool = True
    max_accel: float = 50.0  # Max expected acceleration


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for hockey player tracking.
    
    Uses a coordinated turn motion model that handles:
    - Rapid acceleration/deceleration
    - Direction changes
    - Variable speed movement
    
    State vector: [x, y, w, h, vx, vy, ax, ay, omega]
    - x, y: center position
    - w, h: bounding box size
    - vx, vy: velocity
    - ax, ay: acceleration
    - omega: turn rate (rad/frame)
    """
    
    def __init__(self, config: EKFConfig = None):
        """Initialize EKF with configuration."""
        self.cfg = config or EKFConfig()
        
        # Dimensions
        self.ndim_state = self.cfg.ndim_state
        self.ndim_meas = self.cfg.ndim_meas
        
        # Observation matrix (observe x, y, w, h)
        self._H = np.zeros((self.ndim_meas, self.ndim_state))
        self._H[0, 0] = 1  # x
        self._H[1, 1] = 1  # y
        self._H[2, 2] = 1  # w
        self._H[3, 3] = 1  # h
        
        # Innovation history for adaptive noise
        self._innovation_history = []
        self._max_history = 10
    
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize track from first measurement.
        
        Args:
            measurement: [x, y, w, h] bounding box center format
            
        Returns:
            (mean, covariance): Initial state distribution
        """
        x, y, w, h = measurement
        
        # Initial state: position and size known, velocity/accel unknown
        mean = np.array([
            x, y, w, h,  # Position and size
            0, 0,        # Velocity (unknown)
            0, 0,        # Acceleration (unknown)
            0            # Turn rate (unknown)
        ], dtype=np.float64)
        
        # Initial covariance - high uncertainty for velocity/accel
        std = np.array([
            w * 0.05,    # x (small, we measured it)
            h * 0.05,    # y
            w * 0.1,     # w
            h * 0.1,     # h
            w * 0.5,     # vx (unknown)
            h * 0.5,     # vy
            w * 0.2,     # ax (unknown)
            h * 0.2,     # ay
            0.5          # omega (unknown)
        ])
        
        covariance = np.diag(np.square(std))
        
        return mean, covariance
    
    def _state_transition(self, state: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Non-linear state transition with coordinated turn model.
        
        Args:
            state: Current state [x, y, w, h, vx, vy, ax, ay, omega]
            dt: Time step
            
        Returns:
            Predicted state
        """
        x, y, w, h, vx, vy, ax, ay, omega = state
        
        # Coordinated turn model
        if abs(omega) > 1e-6:
            # Turning motion
            sin_omega = np.sin(omega * dt)
            cos_omega = np.cos(omega * dt)
            
            # Position update with turn
            x_new = x + (vx * sin_omega + vy * (cos_omega - 1)) / omega
            y_new = y + (vy * sin_omega - vx * (cos_omega - 1)) / omega
            
            # Velocity update with turn
            vx_new = vx * cos_omega + vy * sin_omega + ax * dt
            vy_new = vy * cos_omega - vx * sin_omega + ay * dt
        else:
            # Straight line motion
            x_new = x + vx * dt + 0.5 * ax * dt**2
            y_new = y + vy * dt + 0.5 * ay * dt**2
            vx_new = vx + ax * dt
            vy_new = vy + ay * dt
        
        # Size remains relatively constant
        w_new = w
        h_new = h
        
        # Acceleration decays (mean-reverting)
        decay = 0.9
        ax_new = ax * decay
        ay_new = ay * decay
        omega_new = omega * decay
        
        return np.array([x_new, y_new, w_new, h_new, vx_new, vy_new, ax_new, ay_new, omega_new])
    
    def _jacobian_F(self, state: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Compute Jacobian of state transition.
        
        This is the key difference from standard KF - we linearize
        around the current state.
        """
        x, y, w, h, vx, vy, ax, ay, omega = state
        
        F = np.eye(self.ndim_state)
        
        if abs(omega) > 1e-6:
            sin_omega = np.sin(omega * dt)
            cos_omega = np.cos(omega * dt)
            
            # Partial derivatives for turning motion
            # dx/dvx
            F[0, 4] = sin_omega / omega
            # dx/dvy
            F[0, 5] = (cos_omega - 1) / omega
            # dx/domega (complex derivative)
            F[0, 8] = (vx * (omega * dt * cos_omega - sin_omega) + 
                       vy * (-omega * dt * sin_omega - cos_omega + 1)) / (omega**2)
            
            # dy/dvx
            F[1, 4] = -(cos_omega - 1) / omega
            # dy/dvy
            F[1, 5] = sin_omega / omega
            # dy/domega
            F[1, 8] = (vy * (omega * dt * cos_omega - sin_omega) - 
                       vx * (-omega * dt * sin_omega - cos_omega + 1)) / (omega**2)
            
            # dvx/dvx, dvx/dvy
            F[4, 4] = cos_omega
            F[4, 5] = sin_omega
            F[4, 6] = dt  # dvx/dax
            
            # dvy/dvx, dvy/dvy
            F[5, 4] = -sin_omega
            F[5, 5] = cos_omega
            F[5, 7] = dt  # dvy/day
        else:
            # Linear motion Jacobian
            F[0, 4] = dt      # dx/dvx
            F[0, 6] = 0.5 * dt**2  # dx/dax
            F[1, 5] = dt      # dy/dvy
            F[1, 7] = 0.5 * dt**2  # dy/day
            F[4, 6] = dt      # dvx/dax
            F[5, 7] = dt      # dvy/day
        
        # Acceleration decay
        decay = 0.9
        F[6, 6] = decay
        F[7, 7] = decay
        F[8, 8] = decay
        
        return F
    
    def _process_noise(self, state: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Compute adaptive process noise based on motion.
        
        Higher noise for fast-moving objects (more uncertainty).
        """
        x, y, w, h, vx, vy, ax, ay, omega = state
        
        # Base noise
        q_pos = self.cfg.std_pos
        q_vel = self.cfg.std_vel
        q_acc = self.cfg.std_acc
        q_omega = self.cfg.std_omega
        q_size = self.cfg.std_size
        
        # Adaptive: increase noise for faster motion
        if self.cfg.adapt_noise:
            speed = np.sqrt(vx**2 + vy**2)
            accel = np.sqrt(ax**2 + ay**2)
            
            speed_factor = 1.0 + 0.1 * speed / 10.0
            accel_factor = 1.0 + 0.2 * accel / self.cfg.max_accel
            
            q_vel *= speed_factor
            q_acc *= accel_factor
        
        # Process noise covariance
        Q = np.diag([
            (q_pos * w)**2,    # x
            (q_pos * h)**2,    # y
            (q_size * w)**2,   # w
            (q_size * h)**2,   # h
            (q_vel * w)**2,    # vx
            (q_vel * h)**2,    # vy
            (q_acc * w)**2,    # ax
            (q_acc * h)**2,    # ay
            q_omega**2         # omega
        ]) * dt
        
        return Q
    
    def _measurement_noise(self, measurement: np.ndarray) -> np.ndarray:
        """Compute measurement noise covariance."""
        x, y, w, h = measurement
        
        R = np.diag([
            (self.cfg.std_meas_pos * w)**2,   # x
            (self.cfg.std_meas_pos * h)**2,   # y
            (self.cfg.std_meas_size * w)**2,  # w
            (self.cfg.std_meas_size * h)**2   # h
        ])
        
        return R
    
    def predict(
        self, 
        mean: np.ndarray, 
        covariance: np.ndarray,
        dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        EKF prediction step.
        
        Args:
            mean: Current state mean
            covariance: Current state covariance
            dt: Time step
            
        Returns:
            (predicted_mean, predicted_covariance)
        """
        # Non-linear state prediction
        predicted_mean = self._state_transition(mean, dt)
        
        # Linearize around current state
        F = self._jacobian_F(mean, dt)
        
        # Process noise
        Q = self._process_noise(mean, dt)
        
        # Predicted covariance: P' = F * P * F' + Q
        predicted_cov = np.linalg.multi_dot([F, covariance, F.T]) + Q
        
        return predicted_mean, predicted_cov
    
    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        EKF update step with measurement.
        
        Args:
            mean: Predicted state mean
            covariance: Predicted state covariance
            measurement: [x, y, w, h] observation
            
        Returns:
            (updated_mean, updated_covariance)
        """
        # Measurement prediction
        predicted_meas = self._H @ mean
        
        # Innovation (measurement residual)
        innovation = measurement - predicted_meas
        
        # Store for adaptive noise
        self._innovation_history.append(innovation.copy())
        if len(self._innovation_history) > self._max_history:
            self._innovation_history.pop(0)
        
        # Innovation covariance: S = H * P * H' + R
        R = self._measurement_noise(measurement)
        S = np.linalg.multi_dot([self._H, covariance, self._H.T]) + R
        
        # Kalman gain: K = P * H' * S^(-1)
        try:
            chol = scipy.linalg.cho_factor(S)
            K = scipy.linalg.cho_solve(chol, (covariance @ self._H.T).T).T
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            K = covariance @ self._H.T @ np.linalg.pinv(S)
        
        # State update
        updated_mean = mean + K @ innovation
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.ndim_state) - K @ self._H
        updated_cov = (I_KH @ covariance @ I_KH.T + 
                      K @ R @ K.T)
        
        # Ensure size remains positive
        updated_mean[2] = max(updated_mean[2], 1.0)  # w > 0
        updated_mean[3] = max(updated_mean[3], 1.0)  # h > 0
        
        return updated_mean, updated_cov
    
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
            mean: State mean
            covariance: State covariance
            measurements: [N, 4] array of [x, y, w, h]
            only_position: Only use position for gating
            
        Returns:
            Squared Mahalanobis distances
        """
        # Project to measurement space
        predicted_meas = self._H @ mean
        
        # Innovation covariance
        S = self._H @ covariance @ self._H.T
        S += self._measurement_noise(measurements[0] if len(measurements) > 0 else np.array([0, 0, 10, 10]))
        
        if only_position:
            predicted_meas = predicted_meas[:2]
            S = S[:2, :2]
            measurements = measurements[:, :2]
        
        # Mahalanobis distance
        try:
            chol = scipy.linalg.cholesky(S, lower=True)
            diff = measurements - predicted_meas
            z = scipy.linalg.solve_triangular(chol, diff.T, lower=True)
            return np.sum(z * z, axis=0)
        except:
            diff = measurements - predicted_meas
            S_inv = np.linalg.pinv(S)
            return np.array([d @ S_inv @ d for d in diff])
    
    def get_velocity(self, mean: np.ndarray) -> np.ndarray:
        """Get velocity from state."""
        return mean[4:6]
    
    def get_acceleration(self, mean: np.ndarray) -> np.ndarray:
        """Get acceleration from state."""
        return mean[6:8]
    
    def get_bbox(self, mean: np.ndarray) -> np.ndarray:
        """Get bbox [x, y, w, h] from state."""
        return mean[:4]


class InteractingMultipleModel:
    """
    Interacting Multiple Model (IMM) Filter.
    
    Uses multiple motion models and switches between them:
    - Constant velocity (straight skating)
    - Coordinated turn (turning)
    - Constant acceleration (acceleration/deceleration)
    
    Better handles the multi-modal motion in hockey.
    """
    
    def __init__(self, transition_prob: float = 0.05):
        """
        Args:
            transition_prob: Probability of model switching
        """
        # Three motion models
        self.models = {
            'cv': ExtendedKalmanFilter(EKFConfig(std_acc=0.1)),   # Constant velocity
            'ct': ExtendedKalmanFilter(EKFConfig(std_omega=0.3)), # Coordinated turn
            'ca': ExtendedKalmanFilter(EKFConfig(std_acc=1.0))    # Constant acceleration
        }
        
        self.n_models = len(self.models)
        self.model_names = list(self.models.keys())
        
        # Model probabilities
        self.mu = np.ones(self.n_models) / self.n_models
        
        # Transition probability matrix
        p = transition_prob
        self.TPM = np.array([
            [1-2*p, p, p],
            [p, 1-2*p, p],
            [p, p, 1-2*p]
        ])
        
        # State estimates per model
        self.means = {}
        self.covariances = {}
    
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize all models with same measurement."""
        for name, model in self.models.items():
            mean, cov = model.initiate(measurement)
            self.means[name] = mean
            self.covariances[name] = cov
        
        self.mu = np.ones(self.n_models) / self.n_models
        
        # Return combined estimate
        return self._combine_estimates()
    
    def predict(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """IMM prediction step."""
        
        # Step 1: Compute mixing probabilities
        c_bar = self.TPM.T @ self.mu
        mu_ij = np.zeros((self.n_models, self.n_models))
        
        for i in range(self.n_models):
            for j in range(self.n_models):
                mu_ij[i, j] = self.TPM[i, j] * self.mu[i] / (c_bar[j] + 1e-10)
        
        # Step 2: Mixing
        for j, name_j in enumerate(self.model_names):
            mixed_mean = np.zeros_like(self.means[name_j])
            mixed_cov = np.zeros_like(self.covariances[name_j])
            
            for i, name_i in enumerate(self.model_names):
                mixed_mean += mu_ij[i, j] * self.means[name_i]
            
            for i, name_i in enumerate(self.model_names):
                diff = self.means[name_i] - mixed_mean
                mixed_cov += mu_ij[i, j] * (self.covariances[name_i] + np.outer(diff, diff))
            
            self.means[name_j] = mixed_mean
            self.covariances[name_j] = mixed_cov
        
        # Step 3: Model-specific prediction
        for name, model in self.models.items():
            self.means[name], self.covariances[name] = model.predict(
                self.means[name], self.covariances[name], dt
            )
        
        return self._combine_estimates()
    
    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """IMM update step."""
        
        # Model-specific update and likelihood
        likelihoods = np.zeros(self.n_models)
        
        for i, (name, model) in enumerate(self.models.items()):
            # Update
            self.means[name], self.covariances[name] = model.update(
                self.means[name], self.covariances[name], measurement
            )
            
            # Compute likelihood
            likelihoods[i] = self._compute_likelihood(
                model, self.means[name], self.covariances[name], measurement
            )
        
        # Update model probabilities
        c_bar = self.TPM.T @ self.mu
        self.mu = likelihoods * c_bar
        self.mu /= (self.mu.sum() + 1e-10)
        
        return self._combine_estimates()
    
    def _compute_likelihood(
        self,
        model: ExtendedKalmanFilter,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray
    ) -> float:
        """Compute measurement likelihood for a model."""
        predicted_meas = model._H @ mean
        S = model._H @ covariance @ model._H.T + model._measurement_noise(measurement)
        
        innovation = measurement - predicted_meas
        
        try:
            # Gaussian likelihood
            det_S = np.linalg.det(S)
            S_inv = np.linalg.inv(S)
            mahal = innovation @ S_inv @ innovation
            
            likelihood = np.exp(-0.5 * mahal) / np.sqrt((2 * np.pi)**4 * det_S)
            return max(likelihood, 1e-10)
        except:
            return 1e-10
    
    def _combine_estimates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Combine estimates from all models."""
        combined_mean = np.zeros_like(list(self.means.values())[0])
        combined_cov = np.zeros_like(list(self.covariances.values())[0])
        
        for i, name in enumerate(self.model_names):
            combined_mean += self.mu[i] * self.means[name]
        
        for i, name in enumerate(self.model_names):
            diff = self.means[name] - combined_mean
            combined_cov += self.mu[i] * (self.covariances[name] + np.outer(diff, diff))
        
        return combined_mean, combined_cov
    
    def get_active_model(self) -> str:
        """Get most likely active model."""
        return self.model_names[np.argmax(self.mu)]
