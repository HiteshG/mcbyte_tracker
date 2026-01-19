"""
Kalman Filter for Object Tracking
=================================
Optimized Kalman filter implementation for bounding box tracking.
Uses constant velocity model with state [x, y, w, h, vx, vy, vw, vh].
"""

import numpy as np
from typing import Optional, Tuple


class KalmanFilter:
    """
    Kalman filter for tracking bounding boxes in image space.
    
    State vector: [x, y, w, h, vx, vy, vw, vh]
    - (x, y): center position
    - (w, h): width and height  
    - (vx, vy, vw, vh): velocities
    
    Measurement: [x, y, w, h]
    """
    
    # Shared matrices (class-level for efficiency)
    _motion_mat = None
    _update_mat = None
    
    def __init__(self):
        """Initialize Kalman filter with default parameters."""
        ndim, dt = 4, 1.0
        
        # State transition matrix (constant velocity model)
        if KalmanFilter._motion_mat is None:
            KalmanFilter._motion_mat = np.eye(2 * ndim, 2 * ndim)
            for i in range(ndim):
                KalmanFilter._motion_mat[i, ndim + i] = dt
        
        # Observation matrix
        if KalmanFilter._update_mat is None:
            KalmanFilter._update_mat = np.eye(ndim, 2 * ndim)
        
        # Motion and observation uncertainty
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
    
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create track from unassociated measurement.
        
        Args:
            measurement: [x, y, w, h] bounding box center and size
            
        Returns:
            Tuple of (mean, covariance)
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[2],  # x
            2 * self._std_weight_position * measurement[3],  # y
            2 * self._std_weight_position * measurement[2],  # w
            2 * self._std_weight_position * measurement[3],  # h
            10 * self._std_weight_velocity * measurement[2], # vx
            10 * self._std_weight_velocity * measurement[3], # vy
            10 * self._std_weight_velocity * measurement[2], # vw
            10 * self._std_weight_velocity * measurement[3], # vh
        ]
        covariance = np.diag(np.square(std))
        
        return mean, covariance
    
    def predict(
        self, 
        mean: np.ndarray, 
        covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter prediction step.
        
        Args:
            mean: Current state mean
            covariance: Current state covariance
            
        Returns:
            Tuple of (predicted_mean, predicted_covariance)
        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot([
            self._motion_mat, covariance, self._motion_mat.T
        ]) + motion_cov
        
        return mean, covariance
    
    def project(
        self,
        mean: np.ndarray,
        covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state to measurement space.
        
        Args:
            mean: State mean
            covariance: State covariance
            
        Returns:
            Tuple of (projected_mean, projected_covariance)
        """
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        
        innovation_cov = np.diag(np.square(std))
        
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot([
            self._update_mat, covariance, self._update_mat.T
        ])
        
        return mean, covariance + innovation_cov
    
    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter update step.
        
        Args:
            mean: Predicted state mean
            covariance: Predicted state covariance
            measurement: [x, y, w, h] measured bounding box
            
        Returns:
            Tuple of (updated_mean, updated_covariance)
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        
        # Kalman gain
        chol_factor = np.linalg.cholesky(projected_cov)
        kalman_gain = np.linalg.solve(
            chol_factor,
            np.linalg.solve(
                chol_factor,
                np.dot(covariance, self._update_mat.T).T
            ).T
        ).T
        
        # Innovation
        innovation = measurement - projected_mean
        
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot([
            kalman_gain, projected_cov, kalman_gain.T
        ])
        
        return new_mean, new_covariance
    
    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False
    ) -> np.ndarray:
        """
        Compute gating distance (Mahalanobis) between state and measurements.
        
        Args:
            mean: State mean
            covariance: State covariance
            measurements: [N, 4] array of measurements
            only_position: If True, only use position for distance
            
        Returns:
            Array of gating distances
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        
        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurements = measurements[:, :2]
        
        chol = np.linalg.cholesky(projected_cov)
        d = measurements - projected_mean
        z = np.linalg.solve(chol, d.T).T
        
        return np.sum(z * z, axis=1)


class MultiKalmanFilter:
    """
    Batch Kalman filter operations for multiple tracks.
    Optimized for vectorized operations.
    """
    
    def __init__(self):
        self.kf = KalmanFilter()
    
    def multi_predict(
        self,
        means: np.ndarray,
        covariances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict for multiple tracks.
        
        Args:
            means: [N, 8] state means
            covariances: [N, 8, 8] state covariances
            
        Returns:
            Tuple of (predicted_means, predicted_covariances)
        """
        n = means.shape[0]
        new_means = np.zeros_like(means)
        new_covs = np.zeros_like(covariances)
        
        for i in range(n):
            new_means[i], new_covs[i] = self.kf.predict(means[i], covariances[i])
        
        return new_means, new_covs
