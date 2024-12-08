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
            strategy.plot_capital_over_time(self.results[strategy.__class__.__name__], save_path='data/capital_over_time.png')
    
    def graph(self):
        """Visualize the results for each metric in the strategy results."""
        print("Generating graphs...")
        
        for strategy_name, result in self.results.items():
            print(f"Plotting results for {strategy_name}...")
            
            # Plot each column individually, skipping the 'date' column
            for column in result.columns:
                if column != 'date':
                    result.plot(x='date', y=column, title=f"{strategy_name} - {column} Over Time", figsize=(10, 6))
                    plt.xlabel("Date")
                    plt.ylabel(column)
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()

    def metrics(self):
        """Print key metrics of each strategy."""
        print("Calculating metrics...")
        for strategy_name, result in self.results.items():
            print(f"\n{strategy_name} Metrics:")
            print(result)
            resultx = result
        for proportion, strategy in self.strategies:
            print(f"Metrics {strategy.__class__.__name__}")
            x = strategy.metrics(resultx)

    def get_strategy_instance(self, strategy_name):
        """Helper to retrieve the strategy instance by name."""
        for _, strategy in self.strategies:
            if strategy.__class__.__name__ == strategy_name:
                return strategy
        return None

    def plot_pnl(self, save_path=None, log_scale=False):
        """Plot cumulative PnL for each strategy and save the data to CSV."""
        print("Graphing data...")

        for strategy_name, result in self.results.items():
            print(f"\nPlotting {strategy_name} performance...")

            # Get the strategy instance to call its plot method
            strategy_instance = self.get_strategy_instance(strategy_name)
            if strategy_instance:
                # Plot and optionally save the graph
                strategy_instance.plot_pnl(
                    result, 
                    save_path=f"data/{save_path}_{strategy_name}.png" if save_path else None, 
                    log_scale=log_scale  # Pass log_scale as a keyword argument
                )
