"""
modeling.py
===========
Model configuration, training, evaluation, and comparison utilities.

Six regressors are trained and compared:
  1. Linear Regression       4. Gradient Boosting
  2. Elastic Net              5. Support Vector Regression
  3. Random Forest            6. Neural Network (MLP)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def get_model_configs() -> dict:
    """Return the six model configurations with hyperparameter grids."""
    return {
        'Linear Regression': {
            'model': LinearRegression(),
            'scaled': True,
            'params': None,
        },
        'Elastic Net': {
            'model': ElasticNet(max_iter=10000, random_state=42),
            'scaled': True,
            'params': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                       'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]},
        },
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'scaled': False,
            'params': {'n_estimators': [100, 200, 300],
                       'max_depth': [10, 20, 30, None],
                       'min_samples_split': [2, 5, 10]},
        },
        'Gradient Boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'scaled': False,
            'params': {'n_estimators': [100, 200],
                       'learning_rate': [0.01, 0.05, 0.1],
                       'max_depth': [3, 5, 7]},
        },
        'Support Vector Regression': {
            'model': SVR(),
            'scaled': True,
            'params': {'C': [0.1, 1.0, 10.0, 100.0],
                       'epsilon': [0.01, 0.1, 0.2],
                       'kernel': ['rbf', 'linear']},
        },
        'Neural Network': {
            'model': MLPRegressor(max_iter=1000, random_state=42, early_stopping=True),
            'scaled': True,
            'params': {'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                       'alpha': [0.0001, 0.001, 0.01]},
        },
    }


def train_and_evaluate(models_config, X_train, X_test, y_train, y_test,
                        X_train_scaled, X_test_scaled, cv=5, grid_cv=3, verbose=True):
    """
    Train all configured models (with GridSearchCV where applicable),
    compute train/test metrics and 5-fold CV RMSE.

    Returns
    -------
    results : dict of {model_name: {model, train_r2, test_r2, train_rmse,
              test_rmse, test_mae, cv_rmse}}
    predictions : dict of {model_name: {train: array, test: array}}
    """
    results = {}
    predictions = {}

    for name, config in models_config.items():
        if verbose:
            print(f"Training {name}...")

        X_tr = X_train_scaled if config['scaled'] else X_train
        X_te = X_test_scaled if config['scaled'] else X_test

        if config['params']:
            search = GridSearchCV(config['model'], config['params'], cv=grid_cv,
                                   scoring='neg_mean_squared_error', n_jobs=-1)
            search.fit(X_tr, y_train)
            model = search.best_estimator_
            if verbose:
                print(f"  Best params: {search.best_params_}")
        else:
            model = config['model']
            model.fit(X_tr, y_train)

        y_pred_train = model.predict(X_tr)
        y_pred_test = model.predict(X_te)

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)

        cv_scores = cross_val_score(model, X_tr, y_train, cv=cv,
                                     scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())

        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'cv_rmse': cv_rmse,
        }
        predictions[name] = {'train': y_pred_train, 'test': y_pred_test}

        if verbose:
            print(f"  Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f} | "
                  f"Test RMSE: {test_rmse:.4f} | CV RMSE: {cv_rmse:.4f}\n")

    return results, predictions


def results_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert the results dict into a sorted comparison DataFrame."""
    df = pd.DataFrame({
        'Model': list(results.keys()),
        'Train_R2': [results[m]['train_r2'] for m in results],
        'Test_R2': [results[m]['test_r2'] for m in results],
        'Train_RMSE': [results[m]['train_rmse'] for m in results],
        'Test_RMSE': [results[m]['test_rmse'] for m in results],
        'Test_MAE': [results[m]['test_mae'] for m in results],
        'CV_RMSE': [results[m]['cv_rmse'] for m in results],
    })
    return df.sort_values('Test_R2', ascending=False).reset_index(drop=True)


def plot_feature_importance(results: dict, feature_columns: list, output_path: str,
                             tree_models=('Random Forest', 'Gradient Boosting')) -> None:
    """Plot feature importances for the tree-based models."""
    fig, axes = plt.subplots(1, len(tree_models), figsize=(9 * len(tree_models), 7))
    if len(tree_models) == 1:
        axes = [axes]
    fig.suptitle('Feature Importance (Tree-Based Models)', fontsize=16, fontweight='bold')

    for idx, model_name in enumerate(tree_models):
        if model_name not in results:
            continue
        model = results[model_name]['model']
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        sorted_features = [feature_columns[i] for i in order]
        sorted_importances = importances[order]

        axes[idx].barh(range(len(feature_columns)), sorted_importances,
                        alpha=0.85, edgecolor='black', color='teal')
        axes[idx].set_yticks(range(len(feature_columns)))
        axes[idx].set_yticklabels(sorted_features, fontsize=9)
        axes[idx].set_xlabel('Importance', fontweight='bold')
        axes[idx].set_title(model_name, fontweight='bold')
        axes[idx].invert_yaxis()
        axes[idx].grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results: dict, predictions: dict, results_df: pd.DataFrame,
                           y_test, output_path: str) -> None:
    """Create a 9-panel comparison dashboard: metrics + predicted vs actual."""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    x_pos = np.arange(len(results_df))
    width = 0.35

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x_pos - width/2, results_df['Train_R2'], width, label='Train R2', color='steelblue', alpha=0.8)
    ax1.bar(x_pos + width/2, results_df['Test_R2'], width, label='Test R2', color='coral', alpha=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=9)
    ax1.set_title('R2 Score Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x_pos - width/2, results_df['Train_RMSE'], width, label='Train RMSE', color='mediumseagreen', alpha=0.8)
    ax2.bar(x_pos + width/2, results_df['Test_RMSE'], width, label='Test RMSE', color='indianred', alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=9)
    ax2.set_title('RMSE Comparison', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(x_pos - width/2, results_df['Test_MAE'], width, label='Test MAE', color='gold', alpha=0.8)
    ax3.bar(x_pos + width/2, results_df['CV_RMSE'], width, label='CV RMSE', color='mediumpurple', alpha=0.8)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=9)
    ax3.set_title('MAE and CV RMSE', fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    for idx, model_name in enumerate(results.keys()):
        ax = fig.add_subplot(gs[positions[idx]])
        y_pred = predictions[model_name]['test']
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolor='black', linewidth=0.5, s=45)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
        r2 = results[model_name]['test_r2']
        rmse = results[model_name]['test_rmse']
        ax.set_title(f'{model_name}\nR2={r2:.3f}, RMSE={rmse:.3f}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.998)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_residuals(y_test, y_pred, model_name: str, output_path: str) -> None:
    """Plot residual diagnostics: residuals-vs-fitted, histogram, Q-Q plot."""
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Residual Analysis — {model_name}', fontsize=16, fontweight='bold')

    axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolor='black', linewidth=0.5)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted', fontweight='bold')
    axes[0].grid(alpha=0.3)

    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_title('Distribution of Residuals', fontweight='bold')
    axes[1].grid(alpha=0.3)

    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot of Residuals', fontweight='bold')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
