import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load import load_returns_csv
from src.optimizers.equal_weight import get_equal_weight
from src.optimizers.mean_variance import get_mean_variance_weights
from src.optimizers.gmv import get_gmv_weights
from src.optimizers.cvar import get_cvar_weights
from src.backtest.rolling import run_rolling_backtest


def main():
    project_root = Path(__file__).resolve().parents[1]
    print("Project root:", project_root)

    returns_path = project_root / "data" / "processed" / "returns_final.csv"
    base_out_dir = project_root / "data" / "processed" / "backtests"
    os.makedirs(base_out_dir, exist_ok=True)

    returns = load_returns_csv(returns_path)

    strategies = {
        #"cvar": get_cvar_weights,
        "equal_weight": get_equal_weight,
        "mean_variance": get_mean_variance_weights,
        "gmv": get_gmv_weights,
        
    }

    for strategy_name, strategy_func in strategies.items():
        print(f"\nRunning backtest for: {strategy_name}")

        out_dir = base_out_dir / strategy_name
        os.makedirs(out_dir, exist_ok=True)

        portfolio_returns, weights_history = run_rolling_backtest(
            returns=returns,
            strategy_func=strategy_func,
            estimation_window=756,
        )
        
        monthly_portfolio_returns = (
            (1 + portfolio_returns)
            .resample("ME")
            .prod()
            - 1
        )

        portfolio_returns.to_csv(out_dir / (str(strategy_name) + "_daily_portfolio_returns.csv"))
        monthly_portfolio_returns.to_csv(out_dir / (str(strategy_name) + "_monthly_portfolio_returns.csv"))
        weights_history.to_csv(out_dir / (str(strategy_name) + "_weights.csv"))

        print("Backtest completed.")
        print("Portfolio returns shape:", portfolio_returns.shape)
        print("Weights history shape:", weights_history.shape)
        print("Saved to:", out_dir)


if __name__ == "__main__":
    main()