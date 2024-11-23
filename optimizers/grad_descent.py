import autograd.numpy as np

class GradDescentOptimizer:
    def __init__(self, correlation_matrix, capital, penalty_weight=0.1):
        """
        Initialize the Portfolio Optimizer.
        :param correlation_matrix: 2D numpy array representing asset correlations.
        :param capital: Total capital available for allocation.
        :param penalty_weight: Weight for capital utilization penalty.
        """
        self.correlation_matrix = np.array(correlation_matrix)
        self.capital = capital
        self.penalty_weight = penalty_weight

    def tracking_error(self, curr_pos, ideal_pos):
        """
        Calculate the tracking error between the current and ideal positions.
        """
        return np.sqrt(np.sum((curr_pos - ideal_pos) ** 2) / 2)

    def capital_utilization_penalty(self, curr_pos, prices):
        """
        Calculate the penalty for unused capital.
        """
        used_capital = np.sum(curr_pos * prices)
        penalty = max(0, (self.capital - used_capital) / self.capital) ** 2
        return penalty

    def objective_function(self, curr_pos, ideal_pos, prices):
        """
        Calculate the combined objective function.
        """
        te = self.tracking_error(curr_pos, ideal_pos)
        penalty = self.penalty_weight * self.capital_utilization_penalty(curr_pos, prices)
        return te + penalty

    def optimize(self, curr_pos, ideal_pos, prices, learning_rate=0.3, iterations=500):
        """
        Perform gradient descent to optimize positions.
        :param curr_pos: Initial positions (array-like).
        :param ideal_pos: Target positions (array-like).
        :param prices: Asset prices (array-like).
        :param learning_rate: Step size for gradient updates.
        :param iterations: Number of iterations.
        :return: Optimized integer positions.
        """
        curr_pos = ideal_pos # Initialize with small random values
        ideal_pos = np.array(ideal_pos, dtype=np.float64)
        prices = np.array(prices, dtype=np.float64)

        for i in range(iterations):
            # Compute gradients for tracking error and penalty
            te = np.linalg.norm(curr_pos - ideal_pos)
            te_grad = (curr_pos - ideal_pos) / (2 * te) if te != 0 else np.zeros_like(curr_pos)

            used_capital = np.sum(curr_pos * prices)
            penalty_grad = -2 * self.penalty_weight * (self.capital - used_capital) * prices / self.capital

            # Combine and scale gradients
            total_grad = te_grad + penalty_grad
            total_grad /= np.linalg.norm(total_grad) + 1e-8  # Normalize gradients

            # Update positions
            curr_pos -= learning_rate * total_grad
            curr_pos = np.maximum(curr_pos, 0)  # Enforce non-negativity

            # Stop if objective function converges
            if self.objective_function(curr_pos, ideal_pos, prices) < 1e-6:
                break

        # Final rounding to integer positions
        return np.round(curr_pos).astype(int).tolist()
