"""
Flask REST API - Part 2
Provides endpoints for water quality data with filtering and statistics
Enhanced with geographic filtering and time-series resampling
"""

from flask import Flask, jsonify, request
from pymongo import MongoClient
import mongomock
import pandas as pd
from datetime import datetime
from scipy import stats
import numpy as np
import os
from math import radians, cos, sin, asin, sqrt

app = Flask(__name__)

# Database connection (switch between mongomock and MongoDB)
USE_MOCK = True
if USE_MOCK:
    client = mongomock.MongoClient()
else:
    client = MongoClient('mongodb://localhost:27017/')

db = client['water_quality_db']
collection = db['measurements']


def load_data_on_startup():
    """Load cleaned data into database if empty"""
    if collection.count_documents({}) == 0:
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_water_quality.csv')

        if os.path.exists(csv_path):
            print(f"Loading data from {csv_path}...")
            df = pd.read_csv(csv_path)
            records = df.to_dict('records')

            # Convert numpy types to Python types
            for record in records:
                for key, value in record.items():
                    if isinstance(value, (np.integer, np.floating)):
                        record[key] = float(value) if isinstance(value, np.floating) else int(value)
                    elif pd.isna(value):
                        record[key] = None

            collection.insert_many(records)
            print(f"✅ Loaded {len(records)} records into database")
        else:
            print(f"⚠️  CSV file not found: {csv_path}")
            print("Run 'python data_cleaner.py' first to generate data")

# Load data when API starts
load_data_on_startup()


# Helper function to convert MongoDB cursor to list
def serialize_docs(docs):
    """Convert MongoDB documents to JSON-serializable format"""
    result = []
    for doc in docs:
        doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
        result.append(doc)
    return result


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    Returns distance in kilometers
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c  # Radius of earth in kilometers
    return km


@app.route('/')
def home():
    """API home endpoint with documentation"""
    endpoints = {
        'message': 'Water Quality Data API',
        'version': '2.0 - Enhanced with Geographic & Time-Series Features',
        'endpoints': {
            '/api/measurements': 'GET - Retrieve measurements with optional filters',
            '/api/statistics': 'GET - Get statistical summary',
            '/api/outliers': 'GET - Detect outliers in current data',
            '/api/locations': 'GET - List all unique locations',
            '/api/date_range': 'GET - Get min and max dates in dataset',
            '/api/measurements/geographic': 'GET - Filter by geographic area (NEW)',
            '/api/timeseries/resample': 'GET - Resample time series data (NEW)',
            '/api/geographic/bounds': 'GET - Get geographic bounds of data (NEW)'
        },
        'filters': {
            'location_id': 'Filter by location (e.g., ?location_id=LOC001)',
            'start_date': 'Start date (e.g., ?start_date=2023-01-01)',
            'end_date': 'End date (e.g., ?end_date=2023-12-31)',
            'limit': 'Limit number of results (e.g., ?limit=100)'
        }
    }
    return jsonify(endpoints)


@app.route('/api/measurements', methods=['GET'])
def get_measurements():
    """
    Get water quality measurements with optional filters
    Query parameters:
    - location_id: filter by location
    - start_date: filter by start date (YYYY-MM-DD)
    - end_date: filter by end date (YYYY-MM-DD)
    - limit: limit number of results (default: 1000)
    """
    try:
        # Build query filter
        query = {}

        # Location filter
        location_id = request.args.get('location_id')
        if location_id:
            query['location_id'] = location_id

        # Date filters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        if start_date or end_date:
            query['timestamp'] = {}
            if start_date:
                query['timestamp']['$gte'] = start_date
            if end_date:
                query['timestamp']['$lte'] = end_date

        # Get limit
        limit = int(request.args.get('limit', 1000))

        # Query database
        docs = collection.find(query).limit(limit)
        measurements = serialize_docs(docs)

        return jsonify({
            'count': len(measurements),
            'data': measurements
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """
    Get statistical summary of water quality parameters
    Query parameters:
    - location_id: filter by location
    """
    try:
        # Build query
        query = {}
        location_id = request.args.get('location_id')
        if location_id:
            query['location_id'] = location_id

        # Get all matching documents
        docs = list(collection.find(query))

        if not docs:
            return jsonify({'error': 'No data found'}), 404

        # Convert to DataFrame for easy statistics
        df = pd.DataFrame(docs)

        # Calculate statistics for numeric columns
        numeric_cols = ['pH', 'temperature', 'turbidity', 'dissolved_oxygen', 'conductivity']

        stats_dict = {
            'count': len(df),
            'location_id': location_id if location_id else 'all',
            'parameters': {}
        }

        for col in numeric_cols:
            if col in df.columns:
                stats_dict['parameters'][col] = {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'q25': float(df[col].quantile(0.25)),
                    'q75': float(df[col].quantile(0.75))
                }

        return jsonify(stats_dict)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/outliers', methods=['GET'])
def detect_outliers():
    """
    Detect outliers in current dataset using z-score method
    Query parameters:
    - threshold: z-score threshold (default: 3)
    - parameter: specific parameter to check (default: all)
    """
    try:
        threshold = float(request.args.get('threshold', 3))
        parameter = request.args.get('parameter')

        # Get all documents
        docs = list(collection.find())

        if not docs:
            return jsonify({'error': 'No data found'}), 404

        df = pd.DataFrame(docs)

        # Define numeric columns
        numeric_cols = ['pH', 'temperature', 'turbidity', 'dissolved_oxygen', 'conductivity']

        if parameter:
            if parameter not in numeric_cols:
                return jsonify({'error': f'Invalid parameter. Choose from: {numeric_cols}'}), 400
            numeric_cols = [parameter]

        # Calculate z-scores
        z_scores = np.abs(stats.zscore(df[numeric_cols], nan_policy='omit'))

        # Find outliers
        outlier_mask = (z_scores > threshold).any(axis=1)
        outliers_df = df[outlier_mask]

        # Convert to serializable format
        outliers = []
        for idx, row in outliers_df.iterrows():
            outlier_info = {
                '_id': str(row['_id']),
                'timestamp': row['timestamp'],
                'location_id': row['location_id'],
                'values': {}
            }

            for col in numeric_cols:
                outlier_info['values'][col] = float(row[col])

            outliers.append(outlier_info)

        return jsonify({
            'threshold': threshold,
            'parameter': parameter if parameter else 'all',
            'outlier_count': len(outliers),
            'total_records': len(df),
            'percentage': round(len(outliers) / len(df) * 100, 2),
            'outliers': outliers
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/locations', methods=['GET'])
def get_locations():
    """Get list of all unique locations"""
    try:
        locations = collection.distinct('location_id')
        return jsonify({
            'count': len(locations),
            'locations': locations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/date_range', methods=['GET'])
def get_date_range():
    """Get the min and max dates in the dataset"""
    try:
        # Find min and max timestamps
        min_doc = collection.find_one(sort=[('timestamp', 1)])
        max_doc = collection.find_one(sort=[('timestamp', -1)])

        if not min_doc or not max_doc:
            return jsonify({'error': 'No data found'}), 404

        return jsonify({
            'min_date': min_doc['timestamp'],
            'max_date': max_doc['timestamp']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/measurements/geographic', methods=['GET'])
def get_measurements_geographic():
    """
    Get measurements within geographic bounds or radius (EXTRA CREDIT)
    Query parameters:
    - lat: center latitude
    - lon: center longitude
    - radius: radius in km (default: 10)
    OR
    - min_lat, max_lat, min_lon, max_lon: bounding box
    """
    try:
        # Check if radius-based or bounding box
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        radius = request.args.get('radius', type=float, default=10)

        min_lat = request.args.get('min_lat', type=float)
        max_lat = request.args.get('max_lat', type=float)
        min_lon = request.args.get('min_lon', type=float)
        max_lon = request.args.get('max_lon', type=float)

        docs = list(collection.find())
        df = pd.DataFrame(docs)

        if df.empty:
            return jsonify({'error': 'No data found'}), 404

        # Filter by radius
        if lat is not None and lon is not None:
            df['distance'] = df.apply(
                lambda row: haversine(lon, lat, row['longitude'], row['latitude']),
                axis=1
            )
            df_filtered = df[df['distance'] <= radius]

        # Filter by bounding box
        elif all(v is not None for v in [min_lat, max_lat, min_lon, max_lon]):
            df_filtered = df[
                (df['latitude'] >= min_lat) &
                (df['latitude'] <= max_lat) &
                (df['longitude'] >= min_lon) &
                (df['longitude'] <= max_lon)
            ]
        else:
            return jsonify({
                'error': 'Provide either (lat, lon, radius) or (min_lat, max_lat, min_lon, max_lon)'
            }), 400

        # Convert to records
        result = df_filtered.to_dict('records')
        for record in result:
            record['_id'] = str(record['_id'])
            if 'distance' in record:
                record['distance'] = float(record['distance'])

        return jsonify({
            'count': len(result),
            'data': result,
            'filter_type': 'radius' if lat is not None else 'bounding_box'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/timeseries/resample', methods=['GET'])
def resample_timeseries():
    """
    Resample time series data to different frequencies (EXTRA CREDIT)
    Query parameters:
    - frequency: 'H' (hourly), 'D' (daily), 'W' (weekly), 'M' (monthly)
    - aggregation: 'mean', 'median', 'min', 'max', 'sum', 'std'
    - parameter: which parameter to aggregate (default: all numeric)
    - location_id: filter by location (optional)
    - rolling_window: if provided, calculate rolling average (e.g., 7 for 7-period)
    """
    try:
        frequency = request.args.get('frequency', 'D')
        aggregation = request.args.get('aggregation', 'mean')
        parameter = request.args.get('parameter')
        location_id = request.args.get('location_id')
        rolling_window = request.args.get('rolling_window', type=int)

        # Validate inputs
        valid_freq = {'H': 'Hourly', 'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}
        valid_agg = ['mean', 'median', 'min', 'max', 'sum', 'std']

        if frequency not in valid_freq:
            return jsonify({'error': f'Invalid frequency. Use: {list(valid_freq.keys())}'}), 400

        if aggregation not in valid_agg:
            return jsonify({'error': f'Invalid aggregation. Use: {valid_agg}'}), 400

        # Build query
        query = {}
        if location_id:
            query['location_id'] = location_id

        # Get data
        docs = list(collection.find(query))
        if not docs:
            return jsonify({'error': 'No data found'}), 404

        df = pd.DataFrame(docs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Define numeric columns
        numeric_cols = ['pH', 'temperature', 'turbidity', 'dissolved_oxygen', 'conductivity']

        if parameter:
            if parameter not in numeric_cols:
                return jsonify({'error': f'Invalid parameter. Use: {numeric_cols}'}), 400
            numeric_cols = [parameter]

        # Resample
        agg_func = getattr(df[numeric_cols].resample(frequency), aggregation)
        resampled = agg_func()

        # Apply rolling average if requested
        if rolling_window:
            resampled = resampled.rolling(window=rolling_window).mean()
            resampled = resampled.dropna()

        # Convert to records
        resampled_reset = resampled.reset_index()
        result = resampled_reset.to_dict('records')

        # Convert timestamps to strings
        for record in result:
            record['timestamp'] = record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')

        return jsonify({
            'count': len(result),
            'frequency': valid_freq[frequency],
            'aggregation': aggregation,
            'rolling_window': rolling_window,
            'data': result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/geographic/bounds', methods=['GET'])
def get_geographic_bounds():
    """Get the geographic bounds of all data points (EXTRA CREDIT)"""
    try:
        docs = list(collection.find({}, {'latitude': 1, 'longitude': 1}))
        if not docs:
            return jsonify({'error': 'No data found'}), 404

        lats = [doc['latitude'] for doc in docs]
        lons = [doc['longitude'] for doc in docs]

        return jsonify({
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons),
            'center_lat': sum(lats) / len(lats),
            'center_lon': sum(lons) / len(lons)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Check API health and database connection"""
    try:
        count = collection.count_documents({})
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'record_count': count
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)