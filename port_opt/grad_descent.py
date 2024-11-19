import autograd.numpy as np
from autograd import grad

class GradDescentOptimizer:
    def __init__(self, correlation_matrix):
        """
        Initialize the Portfolio Optimizer with the given correlation matrix.
        :param correlation_matrix: 2D numpy array representing asset correlations.
        """
        self.correlation_matrix = np.array(correlation_matrix)
        self.avg_corr_grad = grad(self._average_correlation)

    def tracking_error(self, curr_pos, ideal_pos):
        """
        Calculate the tracking error between the current and ideal positions.
        """
        error = np.sum((curr_pos - ideal_pos) ** 2)
        return np.sqrt(error / 2)

    def _average_correlation(self, positions):
        """
        Calculate the average correlation weighted by the positions.
        """
        total_correlation = 0
        count = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                total_correlation += (
                    self.correlation_matrix[i][j] * positions[i] * positions[j]
                )
                count += 1
        return total_correlation / count if count != 0 else 0

    def objective_function(self, curr_pos, ideal_pos):
        """
        Calculate the combined objective function (tracking error + average correlation).
        """
        te = self.tracking_error(curr_pos, ideal_pos)
        avg_corr = self._average_correlation(curr_pos)
        return te + avg_corr

    def gradient_descent(self, curr_pos, ideal_pos, learning_rate=3, iterations=100):
        """
        Perform gradient descent to optimize the current positions toward the ideal positions.
        :param curr_pos: Initial positions (array-like).
        :param ideal_pos: Target positions (array-like).
        :param learning_rate: Step size for the gradient update.
        :param iterations: Number of iterations for the optimization.
        :return: Optimized integer positions.
        """
        curr_pos = np.array(curr_pos, dtype=np.float64)
        ideal_pos = np.array(ideal_pos, dtype=np.float64)

        for _ in range(iterations):
            # Tracking error derivative component
            te = self.tracking_error(curr_pos, ideal_pos)
            if te != 0:
                te_partial_derivative = (curr_pos - ideal_pos) / (2 * te)
            else:
                te_partial_derivative = np.zeros_like(curr_pos)

            # Average correlation derivative component
            corr_partial_derivative = self.avg_corr_grad(curr_pos)

            # Combined derivative
            total_partial_derivative = te_partial_derivative + corr_partial_derivative

            # Update positions
            curr_pos -= learning_rate * total_partial_derivative

            # Check for convergence
            obj_value = self.objective_function(curr_pos, ideal_pos)
            if obj_value < 1e-6:
                break

        # Round positions to integer values only once at the end
        return np.round(curr_pos).astype(int).tolist()
