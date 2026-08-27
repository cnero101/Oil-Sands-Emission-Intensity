"""
data_processing.py
===================
Data loading, cleaning, and feature engineering functions for the
Alberta Oil Sands GHG Emissions Intensity prediction project.

IMPORTANT: Feature engineering here deliberately EXCLUDES any feature
derived from the emissions variable (Cogen_Adjusted_Emission), since
Emission_Intensity = Cogen_Adjusted_Emission / Production. Including
emissions-derived features would leak the target into the inputs.
See docs/METHODOLOGY.md for a full explanation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(path: str) -> pd.DataFrame:
    """Load the merged facility-level emissions/production dataset."""
    data = pd.read_csv(path)
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove non-operational facility-year records (zero emissions or
    zero production) and drop any remaining missing values.
    """
    data = data.copy()
    data = data[(data['Cogen_Adjusted_Emission'] > 0) & (data['Production'] > 0)]
    data = data.dropna()
    return data.reset_index(drop=True)


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create modeling features WITHOUT using emissions-derived information.

    Feature groups:
      - Production features (volume, log, scale indicators)
      - Temporal features (year, facility age, non-linear trend)
      - Categorical encodings (subsector, technology, product)
      - Facility-level historical features (avg production, years operating)
      - Interaction terms
    """
    data = data.copy()

    # --- Temporal features ---
    data['Years_Since_Start'] = data['Year'] - data['Year'].min()
    data['Facility_Age_Proxy'] = data['Year'] - 2011
    data['Year_Squared'] = data['Year'] ** 2

    # --- Production features (legitimate predictors) ---
    data['Log_Production'] = np.log1p(data['Production'])
    production_median = data['Production'].median()
    production_q75 = data['Production'].quantile(0.75)
    data['High_Production'] = (data['Production'] > production_median).astype(int)
    data['Very_High_Production'] = (data['Production'] > production_q75).astype(int)

    # --- Categorical encoding ---
    categorical_features = ['Subsector', 'Extraction Technology', 'Product']
    encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        data[f'{feature}_Encoded'] = le.fit_transform(data[feature])
        encoders[feature] = le

    # --- Facility-level historical features ---
    data['Facility_Avg_Production'] = data.groupby('Facility')['Production'].transform('mean')
    data['Facility_Years_Operating'] = data.groupby('Facility')['Year'].transform(
        lambda x: x.max() - x.min() + 1
    )

    # --- Interaction terms ---
    data['Tech_Product_Interaction'] = (
        data['Extraction Technology_Encoded'] * data['Product_Encoded']
    )
    data['Subsector_Year_Interaction'] = (
        data['Subsector_Encoded'] * data['Years_Since_Start']
    )

    return data, encoders


# Feature set used for modeling. Deliberately excludes any feature
# derived from Cogen_Adjusted_Emission (target leakage).
FEATURE_COLUMNS = [
    'Production',
    'Log_Production',
    'High_Production',
    'Very_High_Production',
    'Year',
    'Years_Since_Start',
    'Year_Squared',
    'Facility_Age_Proxy',
    'Subsector_Encoded',
    'Extraction Technology_Encoded',
    'Product_Encoded',
    'Facility_Avg_Production',
    'Facility_Years_Operating',
    'Tech_Product_Interaction',
    'Subsector_Year_Interaction',
]

TARGET_COLUMN = 'Emission_Intensity'
