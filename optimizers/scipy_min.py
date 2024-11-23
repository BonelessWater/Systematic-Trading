import cvxpy as cp
import numpy as np

class ScipyOptimizer:
    def __init__(self, float_positions, capital_limit, diversity_constraint, volatility_limit):
        """
        Initialize the Scipy Portfolio Optimizer with constraints and initial float positions.
        """
        self.float_positions = np.array(float_positions)
        self.n = len(self.float_positions)
        self.capital_limit = capital_limit
        self.diversity_constraint = diversity_constraint
        self.volatility_limit = volatility_limit

    def optimize(self):
        """
        Optimize the portfolio to minimize tracking error and meet constraints.
        """
        # Define integer variables
        integer_positions = cp.Variable(self.n, integer=True)

        # Define the objective
        tracking_error = cp.sum_squares(integer_positions - self.float_positions)
        objective = cp.Minimize(tracking_error)

        # Define constraints
        constraints = [
            cp.sum(integer_positions) <= self.capital_limit,
            integer_positions >= self.diversity_constraint,
            cp.norm(integer_positions, 2) <= self.volatility_limit
        ]

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        return {
            'solution': np.round(integer_positions.value).tolist(),
            'tracking_error': problem.value
        }
