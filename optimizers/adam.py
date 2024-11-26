import autograd.numpy as np
from autograd import grad

class AdamOptimizer:
    def __init__(self, correlation_matrix):
        """
        Initialize the optimizer with a correlation matrix.
        """
        self.correlation_matrix = np.array(correlation_matrix)
        assert self.correlation_matrix.shape[0] == self.correlation_matrix.shape[1], "Correlation matrix must be square."
        self.n_assets = self.correlation_matrix.shape[0]  # Number of assets in the correlation matrix

    def tracking_error(self, curr_pos, ideal_pos):
        return np.sum((curr_pos - ideal_pos) ** 2)

    def capital_penalty(self, curr_pos, prices, capital_limit):
        total_capital = np.dot(curr_pos, prices)
        return max(0, total_capital - capital_limit) ** 2

    def diversification_penalty(self, curr_pos):
        return np.sum(curr_pos ** 4)

    def risk_penalty(self, curr_pos):
        """
        Penalize portfolio risk using a non-linear function of the portfolio variance.
        Dynamically align dimensions of the covariance matrix and positions vector.
        """
        n_assets = self.correlation_matrix.shape[0]
        curr_pos = curr_pos[:n_assets]  # Truncate positions to match covariance matrix dimensions
        portfolio_variance = np.dot(curr_pos, np.dot(self.correlation_matrix, curr_pos))
        return portfolio_variance ** 1.5


    def allocation_reward(self, curr_pos):
        return -np.sum(np.exp(-curr_pos))

    def objective_function(self, curr_pos, ideal_pos, prices, capital_limit, lambdas):
        tracking_error = self.tracking_error(curr_pos, ideal_pos)
        capital_penalty = lambdas[0] * self.capital_penalty(curr_pos, prices, capital_limit)
        diversification_penalty = lambdas[1] * self.diversification_penalty(curr_pos)
        risk_penalty = lambdas[2] * self.risk_penalty(curr_pos)
        allocation_reward = lambdas[3] * self.allocation_reward(curr_pos)
        return tracking_error + capital_penalty + diversification_penalty + risk_penalty + allocation_reward

    def optimize(self, ideal_pos, prev_pos, prices, capital_limit, lambdas, learning_rate=0.01, iterations=500, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Perform Adam optimization with consistent alignment of positions and covariance matrix.
        """
        # Align covariance matrix and positions
        n_assets = self.correlation_matrix.shape[0]  # Number of assets in the covariance matrix
        ideal_pos = np.array(ideal_pos[:n_assets], dtype=np.float64)  # Truncate to match covariance matrix dimensions
        prev_pos = np.array(prev_pos[:n_assets], dtype=np.float64)  # Truncate to match
        prices = np.array(prices[:n_assets], dtype=np.float64)  # Truncate to match

        # Initialize Adam optimizer parameters
        curr_pos = prev_pos.copy()
        m = np.zeros_like(curr_pos)
        v = np.zeros_like(curr_pos)
        t = 0

        for _ in range(iterations):
            t += 1

            # Compute gradient of the objective function
            obj_grad = grad(self.objective_function)(curr_pos, ideal_pos, prices, capital_limit, lambdas)

            # Update biased moment estimates
            m = beta1 * m + (1 - beta1) * obj_grad
            v = beta2 * v + (1 - beta2) * (obj_grad ** 2)

            # Compute bias-corrected moment estimates
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            # Update positions
            curr_pos -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            # Debugging output
            obj_value = self.objective_function(curr_pos, ideal_pos, prices, capital_limit, lambdas)
            total_capital = np.dot(curr_pos, prices)
            print(f"Iteration {_}: Objective = {obj_value}, Capital = {total_capital}")

            # Break if objective is no longer improving
            if np.isnan(obj_value) or obj_value < 1e-6:
                print("Converged or encountered instability. Exiting.")
                break

        return np.round(curr_pos).astype(int).tolist()


     
