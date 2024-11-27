import autograd.numpy as np
from autograd import grad

class AdamOptimizer:
    def __init__(self, correlation_matrix):
        """
        Initialize the Adam Portfolio Optimizer with the given correlation matrix.
        :param correlation_matrix: 2D numpy array representing asset correlations.
        """
        self.correlation_matrix = np.array(correlation_matrix)


    def tracking_error(self, curr_pos, ideal_pos):
        """
        Calculate the tracking error between the current and ideal positions.
        """
        error = np.sum((curr_pos - ideal_pos) ** 2)
        return np.sqrt(error / 2)

    def objective_function(self, curr_pos, ideal_pos):
        """
        Calculate the combined objective function (tracking error + average correlation).
        """
        te = self.tracking_error(curr_pos, ideal_pos)
        return te

    def optimize(self, ideal_pos, learning_rate=1, iterations=100, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Perform Adam optimization to optimize the current positions toward the ideal positions.
        """
        curr_pos = np.round(ideal_pos).astype(int).tolist()
        curr_pos = np.array(curr_pos, dtype=np.float64)
        ideal_pos = np.array(ideal_pos, dtype=np.float64)

        # Initialize moment estimates
        m = np.zeros_like(curr_pos)
        v = np.zeros_like(curr_pos)
        t = 0

        for i in range(iterations):
            t += 1

            # Tracking error derivative component
            te = self.tracking_error(curr_pos, ideal_pos)
            if te != 0:
                te_partial_derivative = (curr_pos - ideal_pos) / (2 * te)
            else:
                te_partial_derivative = np.zeros_like(curr_pos)

            # Combined derivative
            total_partial_derivative = te_partial_derivative

            # Update biased moment estimates
            m = beta1 * m + (1 - beta1) * total_partial_derivative
            v = beta2 * v + (1 - beta2) * (total_partial_derivative ** 2)

            # Compute bias-corrected moment estimates
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            # Update positions
            curr_pos -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            # Check for convergence
            obj_value = self.objective_function(curr_pos, ideal_pos)
            if obj_value < 1e-12:
                break

        # Round positions to integer values only once at the end
        return np.round(curr_pos).astype(int).tolist()