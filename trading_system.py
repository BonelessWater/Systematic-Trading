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
            strategy.plot_capital_over_time(self.results[strategy.__class__.__name__], save_path='capital_over_time.png')
    
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

    def get_strategy_instance(self, strategy_name):
        """Helper to retrieve the strategy instance by name."""
        for _, strategy in self.strategies:
            if strategy.__class__.__name__ == strategy_name:
                return strategy
        return None

    def plot_pnl(self, save_path=None, csv_path_prefix='pnl_drawdown', log_scale=False):
        """Plot cumulative PnL for each strategy and save the data to CSV."""
        print("Graphing data...")

        for strategy_name, result in self.results.items():
            print(f"\nPlotting {strategy_name} performance...")

            # Save the data to CSV
            csv_path = f"{csv_path_prefix}_{strategy_name}.csv"
            result.to_csv(csv_path, index=False)
            print(f"Data for {strategy_name} saved to {csv_path}")

            # Get the strategy instance to call its plot method
            strategy_instance = self.get_strategy_instance(strategy_name)
            if strategy_instance:
                # Plot and optionally save the graph
                strategy_instance.plot_pnl(
                    result, 
                    save_path=f"{save_path}_{strategy_name}.png" if save_path else None, 
                    log_scale=log_scale  # Pass log_scale as a keyword argument
                )
