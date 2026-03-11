import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.load import load_returns_csv
from src.optimizers.equal_weight import get_equal_weight
from src.backtest.rolling import run_rolling_backtest


def main():
    project_root = Path(__file__).resolve().parents[1]
    print("Project root:", project_root)

    returns_path = project_root / "data" / "processed" / "returns_final.csv"
    out_dir = project_root / "data" / "processed" / "backtests" / "equal_weight"
    os.makedirs(out_dir, exist_ok=True)

    returns = load_returns_csv(returns_path)

    portfolio_returns, weights_history = run_rolling_backtest(
        returns=returns,
        strategy_func=get_equal_weight,
        estimation_window=756,
    )

    portfolio_returns.to_csv(out_dir / "portfolio_returns.csv")
    weights_history.to_csv(out_dir / "weights.csv")

    print("Backtest completed.")
    print("Portfolio returns shape:", portfolio_returns.shape)
    print("Weights history shape:", weights_history.shape)
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()