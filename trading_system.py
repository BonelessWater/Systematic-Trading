import matplotlib.pyplot as plt

class TradingSystem:
    def __init__(self, strategies):
        """
        Initialize the trading system with a list of strategies.
        Each strategy is a tuple of (proportion of capital, strategy instance).
        """
        self.strategies = strategies
        self.results = {}

    def backtest(self):
        """Run the backtest for each strategy."""
        print("Starting backtest...")
        for proportion, strategy in self.strategies:
            print(f"Backtesting {strategy.__class__.__name__} with {proportion * 100}% of capital.")
            self.results[strategy.__class__.__name__] = strategy.execute()
        return self.results

    def get_strategy_instance(self, strategy_name):
        """Helper to retrieve the strategy instance by name."""
        for _, strategy in self.strategies:
            if strategy.__class__.__name__ == strategy_name:
                return strategy
        return None

