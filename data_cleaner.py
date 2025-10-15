"""
Data Cleaning Script - Part 1
Loads raw water quality CSV, removes outliers using z-score method,
and saves cleaned data to MongoDB/mongomock
"""

import pandas as pd
import numpy as np
from scipy import stats
from pymongo import MongoClient
import mongomock
import json
from datetime import datetime

class WaterQualityETL:
    def __init__(self, use_mock=True):
        """Initialize database connection"""
        if use_mock:
            self.client = mongomock.MongoClient()
        else:
            self.client = MongoClient('mongodb://localhost:27017/')

        self.db = self.client['water_quality_db']
        self.collection = self.db['measurements']

    def load_raw_data(self, filepath):
        """Load raw CSV data"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records")
        return df

    def remove_outliers_zscore(self, df, columns, threshold=3):
        """
        Remove outliers using z-score method

        Parameters:
        - df: DataFrame
        - columns: list of numeric columns to check for outliers
        - threshold: z-score threshold (default: 3)

        Returns:
        - cleaned DataFrame
        - outliers DataFrame
        """
        print(f"\nRemoving outliers using z-score method (threshold={threshold})...")
        df_clean = df.copy()

        # Calculate z-scores for specified columns
        z_scores = np.abs(stats.zscore(df_clean[columns], nan_policy='omit'))

        # Create mask for outliers (any column exceeds threshold)
        outlier_mask = (z_scores > threshold).any(axis=1)

        outliers = df_clean[outlier_mask]
        df_clean = df_clean[~outlier_mask]

        print(f"Removed {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")
        print(f"Remaining records: {len(df_clean)}")

        return df_clean, outliers

    def save_to_database(self, df):
        """Save cleaned data to MongoDB"""
        print("\nSaving to database...")

        # Clear existing data
        self.collection.delete_many({})

        # Convert DataFrame to dict records
        records = df.to_dict('records')

        # Convert numpy types to Python types for MongoDB
        for record in records:
            for key, value in record.items():
                if isinstance(value, (np.integer, np.floating)):
                    record[key] = float(value) if isinstance(value, np.floating) else int(value)
                elif pd.isna(value):
                    record[key] = None

        # Insert into database
        self.collection.insert_many(records)
        print(f"Inserted {len(records)} records into database")

    def get_cleaning_stats(self, original_df, cleaned_df, outliers_df, columns):
        """Generate cleaning statistics"""
        stats_dict = {
            'original_count': len(original_df),
            'cleaned_count': len(cleaned_df),
            'outliers_count': len(outliers_df),
            'removal_percentage': len(outliers_df) / len(original_df) * 100,
            'column_stats': {}
        }

        for col in columns:
            stats_dict['column_stats'][col] = {
                'original_mean': float(original_df[col].mean()),
                'original_std': float(original_df[col].std()),
                'cleaned_mean': float(cleaned_df[col].mean()),
                'cleaned_std': float(cleaned_df[col].std())
            }

        return stats_dict


def generate_sample_data(filepath='data/raw_water_quality.csv', n_samples=1000):
    """Generate sample water quality data"""
    np.random.seed(42)

    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')

    # Generate realistic water quality measurements
    data = {
        'timestamp': dates,
        'location_id': np.random.choice(['LOC001', 'LOC002', 'LOC003', 'LOC004', 'LOC005'], n_samples),
        'pH': np.random.normal(7.2, 0.5, n_samples),
        'temperature': np.random.normal(22, 3, n_samples),  # Celsius
        'turbidity': np.random.gamma(2, 2, n_samples),  # NTU
        'dissolved_oxygen': np.random.normal(8, 1, n_samples),  # mg/L
        'conductivity': np.random.normal(500, 100, n_samples),  # μS/cm
        'latitude': np.random.uniform(40.7, 40.8, n_samples),
        'longitude': np.random.uniform(-74.0, -73.9, n_samples)
    }

    df = pd.DataFrame(data)

    # Add some outliers (5%)
    outlier_indices = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    df.loc[outlier_indices, 'pH'] = np.random.choice([3, 11], len(outlier_indices))
    df.loc[outlier_indices[:10], 'temperature'] = np.random.uniform(50, 60, 10)
    df.loc[outlier_indices[10:20], 'turbidity'] = np.random.uniform(100, 200, 10)

    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Generated sample data: {filepath}")
    return df


def main():
    """Main ETL pipeline"""
    # Create data directory
    import os
    os.makedirs('data', exist_ok=True)

    # Generate or load raw data
    raw_file = 'data/raw_water_quality.csv'
    if not os.path.exists(raw_file):
        generate_sample_data(raw_file)

    # Initialize ETL
    etl = WaterQualityETL(use_mock=True)

    # Load raw data
    df_raw = etl.load_raw_data(raw_file)

    # Define columns to check for outliers
    numeric_columns = ['pH', 'temperature', 'turbidity', 'dissolved_oxygen', 'conductivity']

    # Remove outliers
    df_clean, df_outliers = etl.remove_outliers_zscore(df_raw, numeric_columns, threshold=3)

    # Save cleaned data
    df_clean.to_csv('data/cleaned_water_quality.csv', index=False)
    df_outliers.to_csv('data/outliers_water_quality.csv', index=False)
    print("\nSaved cleaned data to: data/cleaned_water_quality.csv")
    print("Saved outliers to: data/outliers_water_quality.csv")

    # Save to database
    etl.save_to_database(df_clean)

    # Generate and print statistics
    stats = etl.get_cleaning_stats(df_raw, df_clean, df_outliers, numeric_columns)
    print("\n=== Cleaning Statistics ===")
    print(json.dumps(stats, indent=2))

    # Save statistics
    with open('data/cleaning_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("\nSaved statistics to: data/cleaning_stats.json")


if __name__ == "__main__":
    main()