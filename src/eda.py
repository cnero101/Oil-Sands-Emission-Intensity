"""
eda.py
======
Exploratory data analysis: summary statistics and visualizations for
the Alberta Oil Sands GHG Emissions Intensity dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import linregress


def summarize_target(data: pd.DataFrame) -> dict:
    """Compute descriptive statistics for Emission_Intensity."""
    y = data['Emission_Intensity']
    return {
        'mean': y.mean(),
        'median': y.median(),
        'std': y.std(),
        'min': y.min(),
        'q25': y.quantile(0.25),
        'q75': y.quantile(0.75),
        'max': y.max(),
        'skewness': y.skew(),
        'kurtosis': y.kurtosis(),
    }


def temporal_trend(data: pd.DataFrame) -> dict:
    """Fit a linear trend of mean Emission_Intensity over Year."""
    yearly_mean = data.groupby('Year')['Emission_Intensity'].mean()
    slope, intercept, r_value, p_value, std_err = linregress(
        yearly_mean.index, yearly_mean.values
    )
    return {
        'slope': slope,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'yearly_mean': yearly_mean,
    }


def plot_eda_dashboard(data: pd.DataFrame, output_path: str) -> None:
    """
    Generate a 12-panel EDA dashboard covering distribution, category
    comparisons, temporal trends, and diagnostic plots.
    """
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    # 1. Distribution of target
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(data['Emission_Intensity'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(data['Emission_Intensity'].mean(), color='red', linestyle='--', label='Mean')
    ax1.axvline(data['Emission_Intensity'].median(), color='green', linestyle='--', label='Median')
    ax1.set_xlabel('Emissions Intensity (tCO2e/m³)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Emissions Intensity', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Box plot by Subsector
    ax2 = fig.add_subplot(gs[0, 1])
    subsectors = data['Subsector'].unique()
    bp = ax2.boxplot([data[data['Subsector'] == s]['Emission_Intensity'] for s in subsectors],
                      labels=subsectors, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax2.set_title('Intensity by Subsector', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(alpha=0.3, axis='y')

    # 3. Box plot by Technology
    ax3 = fig.add_subplot(gs[0, 2])
    techs = data['Extraction Technology'].unique()
    bp = ax3.boxplot([data[data['Extraction Technology'] == t]['Emission_Intensity'] for t in techs],
                      labels=techs, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
    ax3.set_title('Intensity by Extraction Technology', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(alpha=0.3, axis='y')

    # 4. Temporal trend
    ax4 = fig.add_subplot(gs[1, 0])
    yearly_mean = data.groupby('Year')['Emission_Intensity'].mean()
    yearly_std = data.groupby('Year')['Emission_Intensity'].std()
    ax4.plot(yearly_mean.index, yearly_mean.values, marker='o', linewidth=2.5, color='darkblue')
    ax4.fill_between(yearly_mean.index, yearly_mean - yearly_std, yearly_mean + yearly_std,
                      alpha=0.3, color='lightblue')
    ax4.set_title('Temporal Trend in Emissions Intensity', fontweight='bold')
    ax4.grid(alpha=0.3)

    # 5. Production vs Intensity
    ax5 = fig.add_subplot(gs[1, 1])
    sc = ax5.scatter(np.log1p(data['Production']), data['Emission_Intensity'],
                      alpha=0.6, c=data['Year'], cmap='viridis', edgecolor='black', linewidth=0.5)
    ax5.set_xlabel('Log(Production)')
    ax5.set_ylabel('Emissions Intensity')
    ax5.set_title('Production Volume vs Emissions Intensity', fontweight='bold')
    plt.colorbar(sc, ax=ax5, label='Year')
    ax5.grid(alpha=0.3)

    # 6. Mean intensity by Product
    ax6 = fig.add_subplot(gs[1, 2])
    product_means = data.groupby('Product')['Emission_Intensity'].mean().sort_values(ascending=False)
    ax6.barh(range(len(product_means)), product_means.values,
              color=plt.cm.Set3(np.linspace(0, 1, len(product_means))), edgecolor='black')
    ax6.set_yticks(range(len(product_means)))
    ax6.set_yticklabels(product_means.index)
    ax6.set_title('Mean Intensity by Product', fontweight='bold')
    ax6.grid(alpha=0.3, axis='x')

    # 7. Production vs Total Emissions
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.scatter(data['Production'], data['Cogen_Adjusted_Emission'],
                alpha=0.5, edgecolor='black', linewidth=0.5, color='coral')
    ax7.set_xlabel('Production (m³)')
    ax7.set_ylabel('Total Emissions (tCO2e)')
    ax7.set_title('Production vs Total Emissions', fontweight='bold')
    ax7.grid(alpha=0.3)

    # 8. Violin plot, recent years
    ax8 = fig.add_subplot(gs[2, 1])
    recent_years = sorted(data['Year'].unique())[-5:]
    ax8.violinplot([data[data['Year'] == y]['Emission_Intensity'] for y in recent_years],
                    positions=range(len(recent_years)), showmeans=True, showextrema=True)
    ax8.set_xticks(range(len(recent_years)))
    ax8.set_xticklabels(recent_years)
    ax8.set_title('Recent Years Distribution', fontweight='bold')
    ax8.grid(alpha=0.3, axis='y')

    # 9. Observations by Subsector/Technology
    ax9 = fig.add_subplot(gs[2, 2])
    counts = data.groupby(['Subsector', 'Extraction Technology']).size().unstack(fill_value=0)
    counts.plot(kind='bar', stacked=True, ax=ax9, edgecolor='black', linewidth=0.8)
    ax9.set_title('Observations by Subsector & Technology', fontweight='bold')
    ax9.tick_params(axis='x', rotation=45)
    ax9.legend(title='Technology', fontsize=8)
    ax9.grid(alpha=0.3, axis='y')

    # 10. Q-Q plot
    ax10 = fig.add_subplot(gs[3, 0])
    stats.probplot(data['Emission_Intensity'], dist="norm", plot=ax10)
    ax10.set_title('Q-Q Plot: Emissions Intensity', fontweight='bold')
    ax10.grid(alpha=0.3)

    # 11. Log-transformed distribution
    ax11 = fig.add_subplot(gs[3, 1])
    ax11.hist(np.log1p(data['Emission_Intensity']), bins=50, edgecolor='black',
               alpha=0.7, color='mediumseagreen')
    ax11.set_title('Log-Transformed Emissions Intensity', fontweight='bold')
    ax11.grid(alpha=0.3)

    # 12. Top 10 highest-intensity facilities
    ax12 = fig.add_subplot(gs[3, 2])
    top10 = data.groupby('Facility')['Emission_Intensity'].mean().sort_values(ascending=False).head(10)
    ax12.barh(range(len(top10)), top10.values, color='salmon', edgecolor='black')
    ax12.set_yticks(range(len(top10)))
    ax12.set_yticklabels([f[:28] + '...' if len(f) > 28 else f for f in top10.index], fontsize=8)
    ax12.set_title('Top 10 Facilities by Mean Intensity', fontweight='bold')
    ax12.grid(alpha=0.3, axis='x')

    plt.suptitle('Exploratory Data Analysis — Alberta Oil Sands Emissions Intensity',
                  fontsize=16, fontweight='bold', y=0.998)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation_heatmap(data: pd.DataFrame, numeric_features: list, output_path: str) -> None:
    """Generate a lower-triangle correlation heatmap for numeric features."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, 10))
    corr = data[numeric_features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=1.2, cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1, ax=ax)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
