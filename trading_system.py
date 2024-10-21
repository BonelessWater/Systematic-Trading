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

    def graph(self):
        """Visualize the results."""
        print("Generating graphs...")
        for strategy_name, result in self.results.items():
            result.plot(title=f"{strategy_name} Performance")

    def metrics(self):
        """Print key metrics of each strategy."""
        print("Calculating metrics...")
        for strategy_name, result in self.results.items():
            print(f"\n{strategy_name} Metrics:")
            print(result.describe())
