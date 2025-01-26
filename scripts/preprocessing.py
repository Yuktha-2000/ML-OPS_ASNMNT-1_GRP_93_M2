import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def load_data(filepath):
    return pd.read_csv(filepath)

def get_column_definitions():
    cat_cols = ['Card Type', 'MCC Category', 'Location', 'Device', 'Merchant Reputation', 'Online Transactions Frequency']
    num_cols = ['Amount', 'Previous Transactions', 'Balance Before Transaction', 'Time of Day', 'Velocity', 'Customer Age',
                'Customer Income', 'Card Limit', 'Credit Score', 'Merchant Location History', 'Spending Patterns']
    scale_cols = ['Amount', 'Balance Before Transaction', 'Customer Income', 'Spending Patterns', 'Card Limit', 'Credit Score']
    ordinal_cols = ['Merchant Reputation', 'Online Transactions Frequency']
    reputation_order = ['Bad', 'Average', 'Good']
    transaction_freq_order = ['Low', 'Medium', 'High']
    return cat_cols, num_cols, scale_cols, ordinal_cols, reputation_order, transaction_freq_order

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder


def preprocess_data(df, cat_cols, scale_cols, ordinal_cols, reputation_order, transaction_freq_order):
    """Preprocess the dataset: encode, scale, and handle non-numeric columns."""
    
    # Handle dates
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df.drop(columns=['Date'], inplace=True)

    # Ensure numeric types for all columns
    df = df.apply(pd.to_numeric, errors='ignore')

    # Column transformer for scaling and encoding
    transformer = ColumnTransformer([
        ('one_hot', OneHotEncoder(), cat_cols),
        ('scaler', StandardScaler(), scale_cols),
        ('ordinal_encoder_1', OrdinalEncoder(categories=[reputation_order]), ['Merchant Reputation']),
        ('ordinal_encoder_2', OrdinalEncoder(categories=[transaction_freq_order]), ['Online Transactions Frequency'])
    ], remainder='passthrough')

    X = df.drop(columns=['Is Fraudulent'])  # Features
    y = df['Is Fraudulent']  # Target

    transformed_X = transformer.fit_transform(X)
    feature_names = transformer.get_feature_names_out()
    X = pd.DataFrame(transformed_X, columns=feature_names)

    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)