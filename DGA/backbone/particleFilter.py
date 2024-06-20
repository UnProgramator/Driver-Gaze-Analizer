import numpy as np

class GazeEstimator:
    def __init__(self, num_particles=1000, state_dim=2, motion_std=0.1, measurement_std=0.1):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.motion_std = motion_std
        self.measurement_std = measurement_std
        
        self.particles = np.random.uniform(-1, 1, (self.num_particles, self.state_dim))
        self.weights = np.ones(self.num_particles) / self.num_particles

    def initialize_particles(self):
        self.particles = np.random.uniform(-1, 1, (self.num_particles, self.state_dim))
        self.weights = np.ones(self.num_particles) / self.num_particles

    def predict(self):
        noise = np.random.normal(0, self.motion_std, self.particles.shape)
        self.particles += noise

    def update_weights(self, measurement):
        distances = np.linalg.norm(self.particles - measurement, axis=1)
        self.weights = np.exp(-distances**2 / (2 * self.measurement_std**2))
        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= np.sum(self.weights)  # normalize

    def resample(self):
        indices = np.random.choice(range(len(self.particles)), size=len(self.particles), p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / len(self.weights))

    def estimate(self):
        mean = np.average(self.particles, weights=self.weights, axis=0)
        return mean

    def neff(self):
        return 1. / np.sum(np.square(self.weights))

    def annotate_frame(self, frame, gaze):
        # Function to annotate the frame with the estimated gaze direction
        # This can include drawing arrows or other annotations
        # ...
        return frame

    def filter_gaze(self, pitch, yaw):
        """Apply particle filter to refine the pitch and yaw estimates."""
        measurement = np.array([pitch, yaw])
        
        self.predict()
        self.update_weights(measurement)
        
        if self.neff() < self.num_particles / 2:
            self.resample()
        
        estimated_gaze = self.estimate()
        return estimated_gaze[0], estimated_gaze[1]