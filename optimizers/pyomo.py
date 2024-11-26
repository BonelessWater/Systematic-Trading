from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, minimize, Reals
import numpy as np

class PyomoOptimizer:
    def __init__(self, correlation_matrix):
        """
        Initialize the Pyomo Portfolio Optimizer with the given correlation matrix.
        :param correlation_matrix: 2D numpy array representing asset correlations.
        """
        self.correlation_matrix = np.array(correlation_matrix)

    def tracking_error(self, model, ideal_pos):
        """
        Calculate the tracking error between the current and ideal positions.
        """
        return sum((model.curr_pos[i] - ideal_pos[i]) ** 2 for i in range(len(ideal_pos)))

    def objective_function(self, model, ideal_pos):
        """
        Define the combined objective function (tracking error).
        """
        return self.tracking_error(model, ideal_pos)

    def optimize(self, curr_pos, ideal_pos, prices, capital_limit):
        """
        Perform optimization using Pyomo to optimize the current positions toward the ideal positions,
        with a constraint on total capital.
        
        :param curr_pos: Current positions.
        :param ideal_pos: Ideal positions.
        :param prices: List of asset prices.
        :param capital_limit: Maximum allowable capital for positions.
        :return: Optimized positions.
        """
        n_assets = len(curr_pos)
        model = ConcreteModel()

        # Define variables for current positions with a valid Pyomo domain
        model.curr_pos = Var(range(n_assets), domain=Reals, initialize=lambda model, i: curr_pos[i])

        # Define objective function
        model.obj = Objective(
            rule=lambda model: sum((model.curr_pos[i] - ideal_pos[i]) ** 2 for i in range(n_assets)),
            sense=minimize
        )

        # Add constraints
        # Ensure total positions do not exceed the capital limit
        model.capital_constraint = Constraint(
            expr=sum(model.curr_pos[i] * prices[i] for i in range(n_assets)) <= capital_limit
        )

        # Ensure positions are non-negative (if required)
        model.non_negative_constraint = Constraint(
            expr=sum(model.curr_pos[i] for i in range(n_assets)) >= 0
        )

        # Solve the model
        solver = SolverFactory('ipopt', executable="C:/Users/domdd/Downloads/Ipopt-3.14.16-win64-msvs2019-md/Ipopt-3.14.16-win64-msvs2019-md/bin/ipopt.exe")  # Use a solver suitable for nonlinear optimization
        results = solver.solve(model)

        # Extract optimized positions
        optimized_positions = [model.curr_pos[i].value for i in range(n_assets)]

        # Round positions to integer values only once at the end
        return np.round(optimized_positions).astype(int).tolist()
