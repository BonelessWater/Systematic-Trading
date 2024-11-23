import pyomo.environ as pyo
import numpy as np

class PyomoOptimizer:
    def __init__(self, x_b, cov_matrix, prices, capital, U, max_volatility):
        """
        Initialize the Pyomo Portfolio Optimizer with problem parameters.
        """
        self.x_b = np.array(x_b)
        self.cov_matrix = np.array(cov_matrix)
        self.prices = np.array(prices)
        self.capital = capital
        self.U = U
        self.max_volatility = max_volatility
        self.model = pyo.ConcreteModel()

    def _define_model(self):
        """
        Define the Pyomo model for portfolio optimization.
        """
        n_assets = len(self.x_b)
        self.model.x = pyo.Var(range(n_assets), domain=pyo.Integers, bounds=(0, None))

        # Objective function for tracking error
        def objective_function(model):
            diff = np.array([model.x[i] - self.x_b[i] for i in range(n_assets)])
            return pyo.sqrt(sum(diff[i] * self.cov_matrix[i, j] * diff[j] for i in range(n_assets) for j in range(n_assets)))
        self.model.objective = pyo.Objective(rule=objective_function, sense=pyo.minimize)

        # Constraints
        def capital_constraint(model):
            return sum(self.prices[i] * model.x[i] for i in range(n_assets)) <= self.capital
        self.model.capital_constraint = pyo.Constraint(rule=capital_constraint)

        def volatility_constraint(model):
            return pyo.sqrt(sum(model.x[i] * self.cov_matrix[i, j] * model.x[j] for i in range(n_assets) for j in range(n_assets))) <= self.max_volatility
        self.model.volatility_constraint = pyo.Constraint(rule=volatility_constraint)

        for i in range(n_assets):
            self.model.add_component(f'bound_{i}', pyo.Constraint(expr=self.model.x[i] <= self.U[i]))

    def optimize(self):
        """
        Solve the optimization problem.
        """
        self._define_model()
        solver = pyo.SolverFactory('glpk')
        result = solver.solve(self.model)
        return {
            'solution': [pyo.value(self.model.x[i]) for i in range(len(self.x_b))],
            'objective_value': pyo.value(self.model.objective),
            'status': result.solver.status,
            'termination_condition': result.solver.termination_condition
        }
