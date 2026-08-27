"""
main.py
=======
Entry point for the Alberta Oil Sands GHG Emissions Intensity project.

Runs the full pipeline:
  1. Load and clean data
  2. Engineer features (leakage-free — see docs/METHODOLOGY.md)
  3. Exploratory data analysis (stats + figures)
  4. Train/evaluate six regression models
  5. Save comparison tables and figures to results/

Usage
-----
    python main.py
    python main.py --data data/oil_sands_emissions_merged.csv --output results/

"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_processing import (
    load_data, clean_data, engineer_features, FEATURE_COLUMNS, TARGET_COLUMN
)
from src.eda import summarize_target, temporal_trend, plot_eda_dashboard, plot_correlation_heatmap
from src.modeling import (
    get_model_configs, train_and_evaluate, results_to_dataframe,
    plot_feature_importance, plot_model_comparison, plot_residuals
)

warnings.filterwarnings('ignore')
np.random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description="Oil Sands Emissions Intensity Pipeline")
    parser.add_argument('--data', default='data/oil_sands_emissions_merged.csv',
                         help='Path to the merged input CSV')
    parser.add_argument('--output', default='results/', help='Output directory for results')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    fig_dir = os.path.join(args.output, 'figures')
    table_dir = os.path.join(args.output, 'tables')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Load and clean data
    # ---------------------------------------------------------------
    print("=" * 80)
    print("STEP 1: LOAD & CLEAN DATA")
    print("=" * 80)
    data = load_data(args.data)
    print(f"Loaded {len(data)} raw records")
    data = clean_data(data)
    print(f"Retained {len(data)} operational records after cleaning\n")

    # ---------------------------------------------------------------
    # 2. Feature engineering (leakage-free)
    # ---------------------------------------------------------------
    print("=" * 80)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 80)
    data, encoders = engineer_features(data)
    print(f"Engineered feature set ({len(FEATURE_COLUMNS)} features):")
    for f in FEATURE_COLUMNS:
        print(f"  - {f}")
    print()

    # ---------------------------------------------------------------
    # 3. Exploratory data analysis
    # ---------------------------------------------------------------
    print("=" * 80)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    stats_summary = summarize_target(data)
    print("Emission_Intensity summary:")
    for k, v in stats_summary.items():
        print(f"  {k:10s}: {v:.4f}")

    trend = temporal_trend(data)
    direction = "decreasing" if trend['slope'] < 0 else "increasing"
    print(f"\nTemporal trend: slope={trend['slope']:.4f} ({direction}), "
          f"R2={trend['r_squared']:.3f}, p={trend['p_value']:.3f}")

    plot_eda_dashboard(data, os.path.join(fig_dir, 'eda_dashboard.png'))
    numeric_features = FEATURE_COLUMNS + [TARGET_COLUMN]
    plot_correlation_heatmap(data, numeric_features, os.path.join(fig_dir, 'correlation_heatmap.png'))
    print(f"\nSaved EDA figures to {fig_dir}/\n")

    # ---------------------------------------------------------------
    # 4. Prepare modeling data
    # ---------------------------------------------------------------
    print("=" * 80)
    print("STEP 4: TRAIN / TEST SPLIT & SCALING")
    print("=" * 80)
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    # ---------------------------------------------------------------
    # 5. Train and evaluate models
    # ---------------------------------------------------------------
    print("=" * 80)
    print("STEP 5: MODEL TRAINING & EVALUATION")
    print("=" * 80)
    configs = get_model_configs()
    results, predictions = train_and_evaluate(
        configs, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    )

    results_df = results_to_dataframe(results)
    results_df.to_csv(os.path.join(table_dir, 'model_performance_results.csv'), index=False)

    print("\nFinal ranking (by Test R2):")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]['Model']
    print(f"\nBest model: {best_model_name} "
          f"(Test R2={results_df.iloc[0]['Test_R2']:.4f}, "
          f"Test RMSE={results_df.iloc[0]['Test_RMSE']:.4f})\n")

    # ---------------------------------------------------------------
    # 6. Diagnostics
    # ---------------------------------------------------------------
    print("=" * 80)
    print("STEP 6: FEATURE IMPORTANCE & RESIDUAL DIAGNOSTICS")
    print("=" * 80)
    plot_feature_importance(results, FEATURE_COLUMNS, os.path.join(fig_dir, 'feature_importance.png'))
    plot_model_comparison(results, predictions, results_df, y_test,
                           os.path.join(fig_dir, 'model_comparison.png'))
    plot_residuals(y_test, predictions[best_model_name]['test'], best_model_name,
                   os.path.join(fig_dir, 'residual_analysis.png'))
    print(f"Saved diagnostic figures to {fig_dir}/")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
