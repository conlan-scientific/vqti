import itertools
from pathlib import Path
from typing import List

import pandas as pd

from aroon_simulator import HistoricalData, SignalCalculator, TradingSimulator

def run_simulation(
        hd: HistoricalData,
        aroon_lookback: int = 25,
        aroon_signal_threshold: int = 100,
        max_active_positions: int = 10
    ):
    # Calculate signals based upon Aroon Oscillator
    signal_calc: SignalCalculator = SignalCalculator(
        signal_params={
            "aroon": {
                "p": aroon_lookback,
                "signal_threshold": aroon_signal_threshold
            }
        }
    )
    signal_df: pd.DataFrame = signal_calc.calculate_signals(hd)

    # Simulate historical performance
    sim = TradingSimulator(hd, max_assets = max_active_positions)
    sim.run(signal_df)
    return {
        "aroon_lookback": aroon_lookback,
        "aroon_signal_threshold": aroon_signal_threshold,
        "max_active_positions": max_active_positions,

        "cagr": sim.cagr,
        "volatility": sim.volatility,
        "sharpe_ratio": sim.sharpe_ratio
    }

if __name__ == "__main__":
    eod_data_dir: Path = Path(__file__).parent.parent / "data" / "eod"
    # Load eod CSV files into custom HistoricalData class
    hd: HistoricalData = HistoricalData()
    hd.load_eod_dir(eod_data_dir)

    lookbacks: List = [5, 10, 15, 25, 50, 75, 100, 125, 150, 175, 200]
    thresholds: List = [60] #, 75, 50, 25]
    max_positions: List = [10] #, 20, 50]

    lookbacks: List = [5, 10, 15, 25, 50, 75, 100, 125, 150, 175, 200]
    thresholds: List = [80] #, 75, 50, 25]
    max_positions: List = [10] #, 20, 50]

    lookbacks: List = [5, 10, 15, 25, 50, 75, 100, 125, 150, 175, 200]
    thresholds: List = [40] #, 75, 50, 25]
    max_positions: List = [10] #, 20, 50]

    lookbacks: List = [5, 10, 15, 25, 50, 75, 100, 125, 150, 175, 200]
    thresholds: List = [60] #, 75, 50, 25]
    max_positions: List = [20] #, 20, 50]


    param_combos = list(itertools.product(lookbacks, thresholds, max_positions))
    results = []
    for idx, combo in enumerate(param_combos):
        print(f"Simulation {idx}/{len(param_combos)}... lookback: {combo[0]}, threshold: {combo[1]}, positions: {combo[2]}")
        results.append(
            run_simulation(
                hd,
                aroon_lookback=combo[0],
                aroon_signal_threshold=combo[1],
                max_active_positions=combo[2]
            )
        )
    df = pd.DataFrame(results)
