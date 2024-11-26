import numpy as np

class GreedyOptimizer:
    def __init__(self, correlation_matrix):
        """
        Initialize the Greedy Portfolio Optimizer with the given correlation matrix.
        :param correlation_matrix: 2D numpy array representing asset correlations.
        """
        self.correlation_matrix = np.array(correlation_matrix)

    def tracking_error(self, curr_pos, ideal_pos):
        """
        Calculate the tracking error between current and ideal positions.
        """
        return np.sum((curr_pos - ideal_pos) ** 2)

    def capital_penalty(self, curr_pos, prices, capital_limit):
        """
        Calculate a simple linear penalty for exceeding the capital limit.
        """
        total_capital = np.dot(curr_pos, prices)
        excess_capital = max(0, total_capital - capital_limit)
        return excess_capital

    def objective_function(self, curr_pos, ideal_pos, prices, capital_limit, lambda_cp):
        """
        Simplified objective function: Tracking Error + Capital Penalty.
        """
        tracking_error = self.tracking_error(curr_pos, ideal_pos)
        capital_penalty = lambda_cp * self.capital_penalty(curr_pos, prices, capital_limit)
        return tracking_error + capital_penalty

    def optimize(self, ideal_positions, prev_positions, prices, capital_limit, lambda_cp=1.0, iterations=1000):
        """
        Perform optimization to move current positions toward the ideal positions using a greedy algorithm.
        """
        # Initialize current positions as rounded ideal positions
        curr_positions = np.round(ideal_positions).astype(float)

        # Initialize parameters
        n_assets = len(ideal_positions)
        best_objective_value = self.objective_function(curr_positions, ideal_positions, prices, capital_limit, lambda_cp)

        for _ in range(iterations):
            previous_solution = curr_positions.copy()
            best_idx = None

            for idx in range(n_assets):
                temp_solution = previous_solution.copy()

                # Increment or decrement based on proximity to the ideal position
                if temp_solution[idx] < ideal_positions[idx]:
                    temp_solution[idx] += 1  # Adjust by 1 unit
                else:
                    temp_solution[idx] -= 1  # Adjust by 1 unit

                # Ensure the capital limit is respected
                total_capital = np.dot(temp_solution, prices)
                if total_capital > capital_limit:
                    continue  # Skip this adjustment if it exceeds the capital limit

                # Calculate the new objective value
                obj_value = self.objective_function(temp_solution, ideal_positions, prices, capital_limit, lambda_cp)

                # Update best solution
                if obj_value < best_objective_value:
                    best_objective_value = obj_value
                    best_idx = idx

            # Break if no improvement
            if best_idx is None:
                break

            # Update the solution for the best index
            if curr_positions[best_idx] < ideal_positions[best_idx]:
                curr_positions[best_idx] += 1  # Adjust by 1 unit
            else:
                curr_positions[best_idx] -= 1  # Adjust by 1 unit

        # Round positions to integer values only once at the end
        return np.round(curr_positions).astype(int).tolist()
