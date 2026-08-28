import numpy as np

class PoseKalmanFilter:
    """
    Simple Kalman filter for 6DoF pose prediction.
    Tracks position and velocity, predicts next position.
    """

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.002):
        """
        Args:
            process_noise: How much we expect velocity to change (m/frame)
            measurement_noise: How noisy our pose measurements are (m)
        """
        # State: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6, dtype=np.float64)
        
        # Covariance matrix
        self.P = np.eye(6, dtype=np.float64) * 0.1
        
        # Process noise 
        self.Q = np.eye(6, dtype=np.float64)
        self.Q[:3, :3] *= process_noise ** 2  # Position 
        self.Q[3:, 3:] *= (process_noise * 2) ** 2  # Velocity
        
        # Measurement noise (position only)
        self.R = np.eye(3, dtype=np.float64) * measurement_noise ** 2
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = 1.0  # x += vx
        self.F[1, 4] = 1.0  # y += vy
        self.F[2, 5] = 1.0  # z += vz
        
        # Measurement matrix (only position)
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        
        self._initialized = False
        self._frame_count = 0
       
    
    def initialize(self, position: np.ndarray) -> None:
        self.state[:3] = position
        self.state[3:] = 0.0  # Zero initial velocity
        self.P = np.eye(6, dtype=np.float64) * 0.1
        self._initialized = True
        self._frame_count = 1
    
    def predict(self) -> np.ndarray:
        """
        Advance state one step using the constant-velocity model and
        propagate covariance.
        """
        if not self._initialized:
            return np.zeros(3)

        # x ← Fx, P ← FPFᵀ + Q.
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.state[:3].copy()
    
    def update(self, position: np.ndarray) -> None:
        """
        Update filter with new measured position.
        """
        if not self._initialized:
            self.initialize(position)
            return

        # Standard Kalman correction step (residual → gain → state/covariance update).
        # Measurement residual
        y = position - self.H @ self.state

        # Residual covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
        
        self._frame_count += 1

    def reset(self) -> None:
        """Reset filter state."""
        self.state = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 0.1
        self._initialized = False
        self._frame_count = 0
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized