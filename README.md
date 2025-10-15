# Water Quality Data Pipeline

A complete ETL pipeline for water quality monitoring data, featuring data cleaning, REST API, interactive dashboard, and advanced analytics.

## Overview

This project demonstrates a full-stack data pipeline that processes water quality measurements through multiple stages:

**Pipeline Flow:** Raw CSV → Data Cleaning → MongoDB → REST API → Interactive Dashboard

**Key Features:**
- Z-score based outlier detection and removal
- RESTful API with 9 endpoints
- Interactive Streamlit dashboard with 6 analysis modes
- Geographic filtering (radius and bounding box)
- Time-series resampling and trend analysis
- Professional dark-themed UI with real-time visualizations

## Tech Stack

- **Python 3.10+**
- **Flask 3.1.2** - REST API framework
- **Streamlit 1.50.0** - Interactive dashboard
- **Pandas 2.3.3** - Data manipulation
- **Plotly 6.0.0** - Interactive visualizations
- **MongoDB/mongomock** - NoSQL database
- **SciPy** - Statistical analysis

## Project Structure

### Directory Layout

**Root Directory**
- `README.md` - This documentation
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules
- `data_cleaner.py` - ETL script (Part 1)

**data/** - Generated data files
- `raw_water_quality.csv` - Original dataset with outliers
- `cleaned_water_quality.csv` - Dataset after outlier removal
- `outliers_water_quality.csv` - Removed outlier records
- `cleaning_stats.json` - Statistical summary of cleaning

**api/** - Flask REST API (Part 2)
- `app.py` - Main API server

**client/** - Streamlit Dashboard (Part 3)
- `streamlit_app.py` - Interactive dashboard

### File Descriptions

| File | Location | Purpose |
|------|----------|---------|
| `README.md` | Root | Project documentation |
| `requirements.txt` | Root | Python package dependencies |
| `.gitignore` | Root | Git ignore patterns |
| `data_cleaner.py` | Root | ETL pipeline script |
| `raw_water_quality.csv` | data/ | Original dataset |
| `cleaned_water_quality.csv` | data/ | Cleaned dataset |
| `outliers_water_quality.csv` | data/ | Removed outliers |
| `cleaning_stats.json` | data/ | Cleaning statistics |
| `app.py` | api/ | Flask REST API |
| `streamlit_app.py` | client/ | Streamlit dashboard |

## Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd water-quality-pipeline

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate          # Mac/Linux
venv\Scripts\activate              # Windows

# Install dependencies
pip install -r requirements.txt
2. Run Data Cleaning
'python data_cleaner.py'
What it does:

Generates 1,000 water quality measurements with realistic parameters
Identifies and removes outliers using z-score method (threshold=3)
Saves cleaned data to CSV and loads into MongoDB
Generates statistical summary

Expected output:
Generated sample data: data/raw_water_quality.csv
Loading data from data/raw_water_quality.csv...
Loaded 1000 records

Removing outliers using z-score method (threshold=3)...
Removed 52 outliers (5.20%)
Remaining records: 948

✅ Loaded 948 records into database
Saved cleaned data to: data/cleaned_water_quality.csv
3. Start API Server
# Open new terminal
'cd api
python app.py'
Server runs at: http://localhost:5000
What it does:

Automatically loads cleaned data from CSV into database
Exposes 9 REST endpoints for data access
Provides health check and documentation

4. Launch Dashboard
# Open another terminal
'cd client
streamlit run streamlit_app.py'
Dashboard opens at: http://localhost:8501
What you see:

Real-time connection to API
6 interactive tabs for different analyses
Filters for location, date range, and data limits

API Documentation
Base URL
http://localhost:5000
Endpoints
1. Home / Documentation
httpGET /
Returns API information and available endpoints.
2. Health Check
httpGET /health
Check API status and database connection.
Response:
json{
  "status": "healthy",
  "database": "connected",
  "record_count": 948
}
3. Get Measurements
httpGET /api/measurements
Query Parameters:

location_id - Filter by location (e.g., LOC001)
start_date - Start date (YYYY-MM-DD)
end_date - End date (YYYY-MM-DD)
limit - Max records to return (default: 1000)

Example:
'curl "http://localhost:5000/api/measurements?location_id=LOC001&limit=10"'

Response:
json{
  "count": 10,
  "data": [
    {
      "_id": "...",
      "timestamp": "2023-01-01 00:00:00",
      "location_id": "LOC001",
      "pH": 7.2,
      "temperature": 22.5,
      "turbidity": 3.2,
      "dissolved_oxygen": 8.1,
      "conductivity": 505.3,
      "latitude": 40.75,
      "longitude": -73.95
    }
  ]
}
4. Get Statistics
httpGET /api/statistics
Query Parameters:

location_id - Filter by location (optional)

Example:
'curl "http://localhost:5000/api/statistics?location_id=LOC001"'

Response:
json{
  "count": 190,
  "location_id": "LOC001",
  "parameters": {
    "pH": {
      "mean": 7.21,
      "median": 7.20,
      "std": 0.48,
      "min": 6.01,
      "max": 8.42,
      "q25": 6.89,
      "q75": 7.54
    }
  }
}
5. Detect Outliers
httpGET /api/outliers
Query Parameters:

threshold - Z-score threshold (default: 3)
parameter - Specific parameter to check (optional)

Example:
'curl "http://localhost:5000/api/outliers?threshold=2.5&parameter=pH"'
6. Get Locations
httpGET /api/locations
Returns list of all unique monitoring locations.
7. Get Date Range
httpGET /api/date_range
Returns minimum and maximum dates in the dataset.
8. Geographic Filtering (Extra Credit)
httpGET /api/measurements/geographic
Two filtering modes:
Radius Mode:

lat - Center latitude
lon - Center longitude
radius - Radius in kilometers (default: 10)

Example:
'curl "http://localhost:5000/api/measurements/geographic?lat=40.75&lon=-73.95&radius=5"'
Bounding Box Mode:

min_lat - Minimum latitude
max_lat - Maximum latitude
min_lon - Minimum longitude
max_lon - Maximum longitude

Example:
'curl "http://localhost:5000/api/measurements/geographic?min_lat=40.7&max_lat=40.8&min_lon=-74.0&max_lon=-73.9"'
9. Time-Series Resampling (Extra Credit)
httpGET /api/timeseries/resample
Query Parameters:

frequency - H (hourly), D (daily), W (weekly), M (monthly)
aggregation - mean, median, min, max, sum, std
parameter - Specific parameter (optional, default: all)
location_id - Filter by location (optional)
rolling_window - Apply rolling average (optional)

Example:
# Daily average pH values
curl "http://localhost:5000/api/timeseries/resample?frequency=D&aggregation=mean&parameter=pH"

# Weekly average with 3-period rolling window
curl "http://localhost:5000/api/timeseries/resample?frequency=W&aggregation=mean&rolling_window=3"
10. Get Geographic Bounds
httpGET /api/geographic/bounds
Returns the geographic boundaries of all data points.
Dashboard Features
Tab 1: Overview

Total record count
Summary statistics for all parameters
Parameter distribution histograms with box plots
Detailed statistics table

Tab 2: Time Series

Multi-parameter time series plots
Location-based filtering and comparison
Interactive hover details
Box plots for cross-location analysis

Tab 3: Outliers

Adjustable z-score threshold (2.0-4.0)
Parameter-specific outlier detection
Detailed outlier data table
Scatter plot visualization

Tab 4: Raw Data

Sortable data table
Full dataset explorer
CSV download functionality

Tab 5: Geographic (Extra Credit)

Interactive map of all measurement locations
Radius-based filtering (find points within X km)
Bounding box filtering (rectangular area selection)
Distance calculations using Haversine formula
Location-based statistics

Tab 6: Resampling (Extra Credit)

Multiple time frequencies (Hourly, Daily, Weekly, Monthly)
Various aggregation methods (mean, median, min, max, std)
Rolling average for trend smoothing
Per-parameter or all-parameters analysis
Download resampled data

Data Schema
Water Quality Measurements
FieldTypeUnitDescriptiontimestampdatetime-Measurement timestamplocation_idstring-Location identifier (LOC001-LOC005)pHfloatpH unitsAcidity/alkalinity (typical: 6-8.5)temperaturefloat°CWater temperatureturbidityfloatNTUWater clarity/cloudinessdissolved_oxygenfloatmg/LOxygen content in waterconductivityfloatμS/cmElectrical conductivitylatitudefloatdegreesGPS latitude coordinatelongitudefloatdegreesGPS longitude coordinate
Configuration
Switch to Real MongoDB
To use a real MongoDB instance instead of mongomock:
In data_cleaner.py and api/app.py:
pythonUSE_MOCK = False  # Change from True to False
Requirements:

MongoDB running on mongodb://localhost:27017/

Adjust Outlier Threshold
In data_cleaner.py:
pythondf_clean, df_outliers = etl.remove_outliers_zscore(
    df_raw, 
    numeric_columns, 
    threshold=2.5  # Change from 3 to be more strict
)
Generate More Data
In data_cleaner.py:
pythongenerate_sample_data(
    filepath='data/raw_water_quality.csv', 
    n_samples=5000  # Change from 1000
)
Change API Port
In api/app.py:
pythonapp.run(debug=True, port=8080)  # Change from 5000
Update in client/streamlit_app.py:
pythonAPI_BASE_URL = "http://localhost:8080"

Testing
Test API with curl
# Health check
curl http://localhost:5000/health

# Get 10 measurements
curl "http://localhost:5000/api/measurements?limit=10"

# Get statistics for specific location
curl "http://localhost:5000/api/statistics?location_id=LOC001"

# Find outliers with custom threshold
curl "http://localhost:5000/api/outliers?threshold=2.5"

# Geographic radius search
curl "http://localhost:5000/api/measurements/geographic?lat=40.75&lon=-73.95&radius=10"

# Daily resampling
curl "http://localhost:5000/api/timeseries/resample?frequency=D&aggregation=mean"
Test API with Python
pythonimport requests

# Get measurements
response = requests.get('http://localhost:5000/api/measurements', 
                       params={'limit': 10})
print(response.json())

# Get statistics
response = requests.get('http://localhost:5000/api/statistics')
stats = response.json()
print(f"Average pH: {stats['parameters']['pH']['mean']}")

# Geographic search
response = requests.get('http://localhost:5000/api/measurements/geographic',
                       params={'lat': 40.75, 'lon': -73.95, 'radius': 5})
print(f"Found {response.json()['count']} measurements")
Troubleshooting
"ModuleNotFoundError"
bash# Ensure virtual environment is activated
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows

# Reinstall packages
pip install -r requirements.txt
"Address already in use" (Port 5000)
bash# Find and kill process on Mac/Linux
lsof -i :5000
kill -9 <PID>

# On Windows
netstat -ano | findstr :5000
taskkill /F /PID <PID>
"Cannot connect to API" in Streamlit

Ensure API is running: curl http://localhost:5000/health
Check API terminal for errors
Verify API_BASE_URL in streamlit_app.py matches API port

"No data found" errors

Run data cleaner first: python data_cleaner.py
Check that data/cleaned_water_quality.csv exists
Restart API to reload data

Streamlit not loading
bash# Clear Streamlit cache
streamlit cache clear

# Run with specific port
streamlit run streamlit_app.py --server.port 8502
Key Learning Points
Data Cleaning

Z-score method identifies outliers based on standard deviations from mean
Threshold of 3 captures 99.7% of normal distribution
Removed ~5% of data as outliers for higher quality analysis

REST API Design

Stateless architecture - each request contains all needed info
Query parameters provide flexible filtering without complex URLs
Consistent JSON responses with count and data fields
Error handling returns meaningful messages with proper HTTP codes

Geographic Analysis

Haversine formula calculates distances on a sphere
Radius filtering finds all points within X km of center
Bounding box uses simple lat/lon comparisons for rectangular areas

Time-Series Resampling

Downsampling reduces data points while preserving trends
Aggregation methods (mean, median) handle missing values
Rolling averages smooth noise and highlight patterns

Extra Credit Features
This project implements two extra credit features worth +4 points:
1. Geographic Filters (+2 pts)

Radius-based filtering with distance calculations
Bounding box filtering for rectangular areas
Interactive map visualizations
Location-based statistics

2. Time-Series Resampling (+2 pts)

Multiple frequency options (H, D, W, M)
Multiple aggregation methods
Rolling average smoothing
Trend analysis visualizations

Project Deliverables

 Data cleaning script with z-score outlier detection
 MongoDB/mongomock database integration
 Flask REST API with 9 endpoints
 Streamlit dashboard with 6 tabs
 Interactive Plotly visualizations
 Comprehensive documentation
 requirements.txt with all dependencies
 .gitignore for clean repository
 Geographic filtering (Extra Credit)
 Time-series resampling (Extra Credit)

Technologies Used

Flask - Lightweight REST API framework
Streamlit - Rapid dashboard development
Pandas - Powerful data manipulation
Plotly - Interactive, publication-quality charts
SciPy - Statistical analysis and outlier detection
mongomock - In-memory MongoDB for easy development
Requests - HTTP client for API communication

License
MIT License - Free to use for educational purposes
Authors
[Gustavo Pineda, Luis Delgado, Anders Gutierrez, Jonathan Serrano]
[CIS3590 Internship Ready Software Development]

Acknowledgments

Water quality parameters based on EPA standards
Z-score method follows standard statistical practices
Geographic calculations use the Haversine formula
Project structure follows REST API best practices