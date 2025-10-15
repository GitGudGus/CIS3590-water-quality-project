"""
Streamlit Client - Part 3
Interactive dashboard for water quality data visualization
Enhanced with Geographic Filtering and Time-Series Resampling
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# API Configuration
API_BASE_URL = "http://127.0.0.1:5000"

# Page configuration
st.set_page_config(
    page_title="Water Quality Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Professional Dark Theme with High Contrast
st.markdown("""
    <style>
    /* ========== DARK THEME BASE ========== */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }

    /* ========== SIDEBAR - DARK WITH CONTRAST ========== */
    [data-testid="stSidebar"] {
        background-color: #1a1d24;
        border-right: 2px solid #2d3139;
    }

    /* Sidebar headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 700;
    }

    /* Sidebar labels and text */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #e8e8e8 !important;
        font-weight: 600;
    }

    /* Sidebar selectbox */
    [data-testid="stSidebar"] .stSelectbox label {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 8px;
    }

    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #262b35;
        border: 2px solid #4dabf7;
        border-radius: 8px;
        color: #ffffff !important;
        font-weight: 600;
        font-size: 15px;
    }

    /* CRITICAL: Fix dropdown menu visibility */
    [data-testid="stSidebar"] [data-baseweb="popover"] {
        background-color: #2d3139 !important;
    }

    [data-testid="stSidebar"] [role="listbox"] {
        background-color: #2d3139 !important;
        border: 2px solid #4dabf7;
    }

    [data-testid="stSidebar"] [role="option"] {
        background-color: #2d3139 !important;
        color: #ffffff !important;
        font-weight: 600;
        padding: 12px;
    }

    [data-testid="stSidebar"] [role="option"]:hover {
        background-color: #4dabf7 !important;
        color: #000000 !important;
    }

    /* Sidebar date input */
    [data-testid="stSidebar"] .stDateInput label {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 8px;
    }

    [data-testid="stSidebar"] .stDateInput input {
        background-color: #262b35;
        border: 2px solid #4dabf7;
        border-radius: 8px;
        color: #ffffff !important;
        font-weight: 600;
        font-size: 14px;
        padding: 8px;
    }

    /* Sidebar slider */
    [data-testid="stSidebar"] .stSlider label {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 8px;
    }

    [data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] {
        color: #ffffff !important;
        font-weight: 700;
    }

    [data-testid="stSidebar"] .stSlider [role="slider"] {
        background-color: #4dabf7;
    }

    /* ========== MAIN CONTENT DROPDOWN FIX ========== */
    [data-baseweb="popover"] {
        background-color: #262b35 !important;
        border: 2px solid #4dabf7;
    }

    [role="listbox"] {
        background-color: #262b35 !important;
    }

    [role="option"] {
        background-color: #262b35 !important;
        color: #ffffff !important;
        font-weight: 600;
        padding: 12px;
    }

    [role="option"]:hover {
        background-color: #4dabf7 !important;
        color: #000000 !important;
    }

    /* ========== METRIC CARDS ========== */
    [data-testid="stMetric"] {
        background-color: #1e2329;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #2d3139;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }

    [data-testid="stMetricLabel"] {
        color: #c1c9d2 !important;
        font-weight: 700;
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    [data-testid="stMetricValue"] {
        color: #4dabf7 !important;
        font-size: 32px;
        font-weight: 800;
    }

    [data-testid="stMetricDelta"] {
        font-size: 16px;
        font-weight: 600;
    }

    /* ========== HEADERS ========== */
    h1 {
        color: #ffffff;
        font-weight: 700;
    }

    h2 {
        color: #e8e8e8;
        font-weight: 600;
        margin-top: 1.5rem;
    }

    h3 {
        color: #d0d0d0;
        font-weight: 600;
    }

    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1d24;
        padding: 10px;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #c1c9d2;
        font-weight: 600;
        font-size: 16px;
        padding: 12px 24px;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #4dabf7;
        color: #000000;
    }

    /* ========== TABLES ========== */
    .dataframe {
        font-size: 14px;
        color: #fafafa;
        background-color: #1a1d24;
    }

    .dataframe thead tr th {
        background-color: #262b35 !important;
        color: #ffffff !important;
        font-weight: 700;
        padding: 12px;
        border-bottom: 2px solid #4dabf7;
    }

    .dataframe tbody tr {
        background-color: #1a1d24;
        border-bottom: 1px solid #2d3139;
    }

    .dataframe tbody tr:hover {
        background-color: #262b35;
    }

    .dataframe tbody td {
        color: #e8e8e8;
        padding: 10px;
    }

    /* ========== BUTTONS ========== */
    .stButton button {
        background-color: #4dabf7;
        color: #000000;
        font-weight: 700;
        border-radius: 6px;
        padding: 10px 24px;
        border: none;
        font-size: 16px;
    }

    .stButton button:hover {
        background-color: #74c0fc;
    }

    .stDownloadButton button {
        background-color: #51cf66;
        color: #000000;
        font-weight: 700;
        border-radius: 6px;
        padding: 10px 24px;
    }

    .stDownloadButton button:hover {
        background-color: #69db7c;
    }

    /* ========== ALERTS ========== */
    .stAlert {
        color: #fafafa !important;
        font-weight: 500;
        border-radius: 8px;
        padding: 16px;
        background-color: #262b35;
        border: 2px solid #4dabf7;
    }

    /* ========== INPUT FIELDS (MAIN CONTENT) ========== */
    .stTextInput input, 
    .stSelectbox select,
    .stNumberInput input {
        background-color: #262b35;
        color: #ffffff !important;
        border: 2px solid #4dabf7;
        border-radius: 6px;
        padding: 8px;
        font-size: 14px;
        font-weight: 600;
    }

    .stTextInput label,
    .stSelectbox label,
    .stNumberInput label {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 16px;
    }

    /* Multi-select */
    .stMultiSelect label {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 16px;
    }

    .stMultiSelect [data-baseweb="tag"] {
        background-color: #4dabf7;
        color: #000000;
        font-weight: 600;
    }

    .stMultiSelect > div > div {
        background-color: #262b35;
        border: 2px solid #4dabf7;
    }

    /* ========== SLIDER (MAIN CONTENT) ========== */
    .stSlider label {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 16px;
    }

    .stSlider [data-testid="stTickBar"] {
        color: #ffffff !important;
        font-weight: 700;
    }

    .stSlider [role="slider"] {
        background-color: #4dabf7;
    }

    /* ========== RADIO BUTTONS ========== */
    .stRadio label {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 16px;
    }

    .stRadio [role="radiogroup"] label {
        color: #e8e8e8 !important;
        font-weight: 600;
    }

    /* ========== CHECKBOX ========== */
    .stCheckbox label {
        color: #e8e8e8 !important;
        font-weight: 600;
    }

    /* ========== GENERAL TEXT ========== */
    p, span, div, label {
        color: #e8e8e8;
    }

    /* ========== PLOTLY CHARTS - DARK THEME ========== */
    .js-plotly-plot {
        background-color: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_measurements(location_id=None, start_date=None, end_date=None, limit=1000):
    """Fetch measurements from API"""
    try:
        params = {'limit': limit}
        if location_id:
            params['location_id'] = location_id
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date

        response = requests.get(f"{API_BASE_URL}/api/measurements", params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching measurements: {e}")
        return None


def get_statistics(location_id=None):
    """Fetch statistics from API"""
    try:
        params = {}
        if location_id:
            params['location_id'] = location_id

        response = requests.get(f"{API_BASE_URL}/api/statistics", params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching statistics: {e}")
        return None


def get_outliers(threshold=3, parameter=None):
    """Fetch outliers from API"""
    try:
        params = {'threshold': threshold}
        if parameter:
            params['parameter'] = parameter

        response = requests.get(f"{API_BASE_URL}/api/outliers", params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching outliers: {e}")
        return None


def get_locations():
    """Fetch available locations from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/locations")
        response.raise_for_status()
        return response.json()['locations']
    except Exception as e:
        st.error(f"Error fetching locations: {e}")
        return []


def get_date_range():
    """Fetch date range from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/date_range")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None


def get_measurements_geographic(lat=None, lon=None, radius=None,
                                min_lat=None, max_lat=None, min_lon=None, max_lon=None):
    """Fetch measurements with geographic filters (EXTRA CREDIT)"""
    try:
        params = {}
        if lat is not None and lon is not None:
            params['lat'] = lat
            params['lon'] = lon
            params['radius'] = radius or 10
        elif all(v is not None for v in [min_lat, max_lat, min_lon, max_lon]):
            params['min_lat'] = min_lat
            params['max_lat'] = max_lat
            params['min_lon'] = min_lon
            params['max_lon'] = max_lon

        response = requests.get(f"{API_BASE_URL}/api/measurements/geographic", params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching geographic data: {e}")
        return None


def get_timeseries_resampled(frequency='D', aggregation='mean', parameter=None,
                            location_id=None, rolling_window=None):
    """Fetch resampled time series data (EXTRA CREDIT)"""
    try:
        params = {
            'frequency': frequency,
            'aggregation': aggregation
        }
        if parameter:
            params['parameter'] = parameter
        if location_id:
            params['location_id'] = location_id
        if rolling_window:
            params['rolling_window'] = rolling_window

        response = requests.get(f"{API_BASE_URL}/api/timeseries/resample", params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching resampled data: {e}")
        return None


def get_geographic_bounds():
    """Fetch geographic bounds (EXTRA CREDIT)"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/geographic/bounds")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None


def main():
    """Main Streamlit application"""

    # Header
    st.title("💧 Water Quality Dashboard")
    st.markdown("Interactive visualization of water quality measurements with geographic and time-series analysis")

    # Check API connection
    if not check_api_health():
        st.error("⚠️ Cannot connect to API. Please ensure the Flask API is running on http://localhost:5000")
        st.info("Run: `python api/app.py` in a separate terminal")
        st.stop()

    st.success("✅ Connected to API")

    # Sidebar filters
    st.sidebar.header("Filters")

    # Get available locations
    locations = get_locations()
    location_filter = st.sidebar.selectbox(
        "Location",
        ["All"] + locations
    )

    # Date range filter
    date_range_data = get_date_range()
    if date_range_data:
        min_date = datetime.strptime(date_range_data['min_date'], '%Y-%m-%d %H:%M:%S')
        max_date = datetime.strptime(date_range_data['max_date'], '%Y-%m-%d %H:%M:%S')

        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date()
        )
    else:
        date_range = None

    # Data limit
    data_limit = st.sidebar.slider("Max Records", 100, 5000, 1000, 100)

    # Tabs - NOW WITH 6 TABS INCLUDING EXTRA CREDIT FEATURES
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "📈 Time Series",
        "🔍 Outliers",
        "📋 Raw Data",
        "🗺️ Geographic",
        "📉 Resampling"
    ])

    # Prepare filters
    loc_filter = location_filter if location_filter != "All" else None
    start_date_str = date_range[0].strftime('%Y-%m-%d') if date_range and len(date_range) > 0 else None
    end_date_str = date_range[1].strftime('%Y-%m-%d') if date_range and len(date_range) > 1 else None

    # Fetch data
    measurements_data = get_measurements(loc_filter, start_date_str, end_date_str, data_limit)
    stats_data = get_statistics(loc_filter)

    if not measurements_data or not measurements_data['data']:
        st.warning("No data available for the selected filters")
        return

    df = pd.DataFrame(measurements_data['data'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Tab 1: Overview with Statistics
    with tab1:
        st.header("Statistical Overview")

        # Display record count
        st.metric("Total Records", f"{measurements_data['count']:,}")

        # Statistics table
        if stats_data and 'parameters' in stats_data:
            st.subheader("Summary Statistics")

            # Create metrics in columns
            params = list(stats_data['parameters'].keys())
            cols = st.columns(len(params))

            for idx, param in enumerate(params):
                with cols[idx]:
                    param_stats = stats_data['parameters'][param]
                    st.metric(
                        label=param.replace('_', ' ').title(),
                        value=f"{param_stats['mean']:.2f}",
                        delta=f"±{param_stats['std']:.2f}"
                    )

            # Detailed statistics table
            st.subheader("Detailed Statistics")
            stats_df = pd.DataFrame(stats_data['parameters']).T
            st.dataframe(stats_df.style.format("{:.3f}"), use_container_width=True)

        # Distribution plots
        st.subheader("Parameter Distributions")
        numeric_cols = ['pH', 'temperature', 'turbidity', 'dissolved_oxygen', 'conductivity']

        col1, col2 = st.columns(2)

        for idx, param in enumerate(numeric_cols):
            if param in df.columns:
                with col1 if idx % 2 == 0 else col2:
                    fig = px.histogram(
                        df,
                        x=param,
                        title=f"{param.replace('_', ' ').title()} Distribution",
                        nbins=30,
                        marginal="box"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Time Series
    with tab2:
        st.header("Time Series Analysis")

        # Parameter selector
        param_options = ['pH', 'temperature', 'turbidity', 'dissolved_oxygen', 'conductivity']
        selected_params = st.multiselect(
            "Select Parameters",
            param_options,
            default=['pH', 'temperature']
        )

        if selected_params:
            # Single plot with multiple traces
            fig = go.Figure()

            for param in selected_params:
                if param in df.columns:
                    if location_filter == "All":
                        # Group by location
                        for location in df['location_id'].unique():
                            loc_df = df[df['location_id'] == location]
                            fig.add_trace(go.Scatter(
                                x=loc_df['timestamp'],
                                y=loc_df[param],
                                mode='lines',
                                name=f"{param} - {location}",
                                line=dict(width=2)
                            ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=df['timestamp'],
                            y=df[param],
                            mode='lines+markers',
                            name=param,
                            line=dict(width=2)
                        ))

            fig.update_layout(
                title="Water Quality Parameters Over Time",
                xaxis_title="Timestamp",
                yaxis_title="Value",
                hovermode='x unified',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        # Location comparison
        if location_filter == "All":
            st.subheader("Location Comparison")

            comparison_param = st.selectbox(
                "Select Parameter for Comparison",
                param_options
            )

            if comparison_param in df.columns:
                fig = px.box(
                    df,
                    x='location_id',
                    y=comparison_param,
                    title=f"{comparison_param.replace('_', ' ').title()} by Location",
                    color='location_id'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Outliers
    with tab3:
        st.header("Outlier Detection")

        col1, col2 = st.columns([1, 3])

        with col1:
            threshold = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.1)
            outlier_param = st.selectbox(
                "Parameter",
                ["All"] + param_options
            )

        # Fetch outliers
        outlier_param_filter = None if outlier_param == "All" else outlier_param
        outliers_data = get_outliers(threshold, outlier_param_filter)

        if outliers_data:
            with col2:
                st.metric("Outliers Found", outliers_data['outlier_count'])
                st.metric("Percentage", f"{outliers_data['percentage']}%")

            if outliers_data['outliers']:
                st.subheader("Outlier Details")
                outliers_df = pd.DataFrame(outliers_data['outliers'])

                # Expand values column
                if 'values' in outliers_df.columns:
                    values_df = pd.json_normalize(outliers_df['values'])
                    outliers_display = pd.concat([
                        outliers_df[['timestamp', 'location_id']],
                        values_df
                    ], axis=1)

                    st.dataframe(outliers_display, use_container_width=True)

                    # Visualization
                    if outlier_param != "All" and outlier_param in values_df.columns:
                        fig = px.scatter(
                            outliers_display,
                            x='timestamp',
                            y=outlier_param,
                            color='location_id',
                            title=f"Outliers: {outlier_param}",
                            size_max=10
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No outliers detected with current threshold")

    # Tab 4: Raw Data
    with tab4:
        st.header("Raw Data Explorer")

        st.dataframe(
            df.sort_values('timestamp', ascending=False),
            use_container_width=True,
            height=500
        )

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"water_quality_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    # Tab 5: Geographic Filters (EXTRA CREDIT +2 pts)
    with tab5:
        st.header("🗺️ Geographic Analysis")

        st.markdown("""
        Filter water quality measurements by geographic location using either:
        - **Radius**: Find all measurements within X kilometers of a center point
        - **Bounding Box**: Define a rectangular area using latitude/longitude bounds
        """)

        # Get geographic bounds
        bounds = get_geographic_bounds()

        if bounds:
            st.subheader("Dataset Geographic Bounds")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min Latitude", f"{bounds['min_lat']:.4f}°")
            with col2:
                st.metric("Max Latitude", f"{bounds['max_lat']:.4f}°")
            with col3:
                st.metric("Min Longitude", f"{bounds['min_lon']:.4f}°")
            with col4:
                st.metric("Max Longitude", f"{bounds['max_lon']:.4f}°")

            # Map visualization
            st.subheader("All Measurement Locations")

            # Create map with all points
            fig_map = px.scatter_mapbox(
                df,
                lat='latitude',
                lon='longitude',
                color='location_id',
                size='pH',
                hover_data=['pH', 'temperature', 'turbidity', 'dissolved_oxygen'],
                zoom=11,
                height=500,
                title="Water Quality Measurement Locations"
            )

            fig_map.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(
                    center=dict(
                        lat=bounds['center_lat'],
                        lon=bounds['center_lon']
                    )
                )
            )

            st.plotly_chart(fig_map, use_container_width=True)

            # Geographic filtering options
            st.subheader("Filter by Geographic Area")

            filter_type = st.radio(
                "Filter Type",
                ["Radius", "Bounding Box"],
                horizontal=True
            )

            if filter_type == "Radius":
                st.markdown("**Find measurements within a radius of a center point**")

                col1, col2, col3 = st.columns(3)

                with col1:
                    center_lat = st.number_input(
                        "Center Latitude",
                        value=float(bounds['center_lat']),
                        format="%.6f",
                        help="Latitude of the center point"
                    )
                with col2:
                    center_lon = st.number_input(
                        "Center Longitude",
                        value=float(bounds['center_lon']),
                        format="%.6f",
                        help="Longitude of the center point"
                    )
                with col3:
                    radius_km = st.slider(
                        "Radius (km)",
                        min_value=1,
                        max_value=50,
                        value=10,
                        help="Search radius in kilometers"
                    )

                if st.button("🔍 Apply Radius Filter", type="primary"):
                    with st.spinner("Filtering measurements..."):
                        geo_data = get_measurements_geographic(
                            lat=center_lat,
                            lon=center_lon,
                            radius=radius_km
                        )

                        if geo_data and geo_data['data']:
                            st.success(f"✅ Found {geo_data['count']} measurements within {radius_km}km")
                            geo_df = pd.DataFrame(geo_data['data'])

                            # Show filtered points on map
                            fig_filtered = px.scatter_mapbox(
                                geo_df,
                                lat='latitude',
                                lon='longitude',
                                color='distance',
                                size='pH',
                                hover_data=['distance', 'pH', 'temperature', 'location_id'],
                                zoom=11,
                                height=400,
                                title=f"Measurements within {radius_km}km (colored by distance)",
                                color_continuous_scale="Viridis"
                            )

                            fig_filtered.update_layout(mapbox_style="open-street-map")
                            st.plotly_chart(fig_filtered, use_container_width=True)

                            # Statistics
                            st.subheader("Statistics by Location")
                            location_stats = geo_df.groupby('location_id').agg({
                                'pH': 'mean',
                                'temperature': 'mean',
                                'distance': 'mean'
                            }).round(3)
                            st.dataframe(location_stats, use_container_width=True)

                            # Show sample data
                            st.subheader("Sample Data")
                            st.dataframe(
                                geo_df[['timestamp', 'location_id', 'distance', 'pH', 'temperature', 'turbidity']].head(20),
                                use_container_width=True
                            )
                        else:
                            st.warning("⚠️ No measurements found in this area")

            else:  # Bounding Box
                st.markdown("**Define a rectangular area using latitude/longitude bounds**")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Southwest Corner**")
                    box_min_lat = st.number_input(
                        "Min Latitude",
                        value=float(bounds['min_lat']),
                        format="%.6f"
                    )
                    box_min_lon = st.number_input(
                        "Min Longitude",
                        value=float(bounds['min_lon']),
                        format="%.6f"
                    )

                with col2:
                    st.markdown("**Northeast Corner**")
                    box_max_lat = st.number_input(
                        "Max Latitude",
                        value=float(bounds['max_lat']),
                        format="%.6f"
                    )
                    box_max_lon = st.number_input(
                        "Max Longitude",
                        value=float(bounds['max_lon']),
                        format="%.6f"
                    )

                if st.button("🔍 Apply Bounding Box Filter", type="primary"):
                    with st.spinner("Filtering measurements..."):
                        geo_data = get_measurements_geographic(
                            min_lat=box_min_lat,
                            max_lat=box_max_lat,
                            min_lon=box_min_lon,
                            max_lon=box_max_lon
                        )

                        if geo_data and geo_data['data']:
                            st.success(f"✅ Found {geo_data['count']} measurements in bounding box")
                            geo_df = pd.DataFrame(geo_data['data'])

                            # Show filtered points on map
                            fig_box = px.scatter_mapbox(
                                geo_df,
                                lat='latitude',
                                lon='longitude',
                                color='location_id',
                                size='pH',
                                hover_data=['pH', 'temperature', 'location_id'],
                                zoom=11,
                                height=400,
                                title="Measurements in Bounding Box"
                            )

                            fig_box.update_layout(mapbox_style="open-street-map")
                            st.plotly_chart(fig_box, use_container_width=True)

                            # Statistics by location
                            st.subheader("Statistics by Location")
                            location_stats = geo_df.groupby('location_id')[['pH', 'temperature', 'turbidity']].mean().round(3)
                            st.dataframe(location_stats, use_container_width=True)
                        else:
                            st.warning("⚠️ No measurements found in this area")

    # Tab 6: Time Series Resampling (EXTRA CREDIT +2 pts)
    with tab6:
        st.header("📉 Time Series Resampling & Trends")

        st.markdown("""
        Resample water quality data to different time frequencies and analyze trends over time.
        This is useful for identifying patterns and reducing noise in the data.
        """)

        # Resampling options
        col1, col2, col3 = st.columns(3)

        with col1:
            resample_freq = st.selectbox(
                "Frequency",
                options=['H', 'D', 'W', 'M'],
                format_func=lambda x: {'H': 'Hourly', 'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}[x],
                index=1,  # Default to Daily
                help="Time frequency for aggregation"
            )

        with col2:
            resample_agg = st.selectbox(
                "Aggregation",
                options=['mean', 'median', 'min', 'max', 'std'],
                index=0,
                help="Statistical method for aggregation"
            )

        with col3:
            resample_param = st.selectbox(
                "Parameter",
                options=['All'] + ['pH', 'temperature', 'turbidity', 'dissolved_oxygen', 'conductivity'],
                help="Water quality parameter to analyze"
            )

        # Rolling average option
        use_rolling = st.checkbox("Apply Rolling Average", help="Smooth data using a moving average")
        rolling_window = None

        if use_rolling:
            rolling_window = st.slider(
                "Rolling Window Size",
                min_value=2,
                max_value=30,
                value=7,
                help="Number of periods for rolling average calculation"
            )

        # Location filter for resampling
        resample_location = st.selectbox(
            "Filter by Location (Optional)",
            ["All"] + locations,
            key="resample_location"
        )

        if st.button("📊 Resample Data", type="primary"):
            param_filter = None if resample_param == "All" else resample_param
            loc_filter_resample = None if resample_location == "All" else resample_location

            with st.spinner("Resampling time series data..."):
                resampled_data = get_timeseries_resampled(
                    frequency=resample_freq,
                    aggregation=resample_agg,
                    parameter=param_filter,
                    location_id=loc_filter_resample,
                    rolling_window=rolling_window
                )

                if resampled_data and resampled_data['data']:
                    st.success(f"✅ Successfully resampled to {resampled_data['frequency']} frequency using {resample_agg}")

                    # Display info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Data Points", resampled_data['count'])
                    with col2:
                        st.metric("Frequency", resampled_data['frequency'])
                    with col3:
                        st.metric("Aggregation", resampled_data['aggregation'].title())

                    # Convert to DataFrame
                    resampled_df = pd.DataFrame(resampled_data['data'])
                    resampled_df['timestamp'] = pd.to_datetime(resampled_df['timestamp'])

                    # Plot resampled data
                    st.subheader("Resampled Time Series Visualization")

                    if resample_param == "All":
                        # Plot all parameters
                        params_to_plot = ['pH', 'temperature', 'turbidity', 'dissolved_oxygen', 'conductivity']

                        for param in params_to_plot:
                            if param in resampled_df.columns:
                                fig = px.line(
                                    resampled_df,
                                    x='timestamp',
                                    y=param,
                                    title=f"{param.replace('_', ' ').title()} - {resampled_data['frequency']} {resample_agg.title()}",
                                    markers=True
                                )
                                fig.update_traces(line_color='#4dabf7', line_width=3)
                                fig.update_layout(
                                    height=300,
                                    template="plotly_dark",
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Plot single parameter
                        fig = px.line(
                            resampled_df,
                            x='timestamp',
                            y=resample_param,
                            title=f"{resample_param.replace('_', ' ').title()} - {resampled_data['frequency']} {resample_agg.title()}",
                            markers=True
                        )

                        if rolling_window:
                            fig.update_layout(
                                title=f"{resample_param.title()} - {resampled_data['frequency']} {resample_agg.title()} (Rolling Avg: {rolling_window})"
                            )

                        fig.update_traces(line_color='#4dabf7', line_width=3, marker_size=8)
                        fig.update_layout(
                            height=500,
                            template="plotly_dark",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Show data table
                    st.subheader("Resampled Data Table")
                    st.dataframe(resampled_df, use_container_width=True, height=300)

                    # Download resampled data
                    csv_resampled = resampled_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Resampled Data",
                        data=csv_resampled,
                        file_name=f"resampled_{resample_freq}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

                    # Summary statistics
                    if resample_param != "All":
                        st.subheader("Summary Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean", f"{resampled_df[resample_param].mean():.2f}")
                        with col2:
                            st.metric("Median", f"{resampled_df[resample_param].median():.2f}")
                        with col3:
                            st.metric("Min", f"{resampled_df[resample_param].min():.2f}")
                        with col4:
                            st.metric("Max", f"{resampled_df[resample_param].max():.2f}")
                else:
                    st.error("❌ Failed to resample data. Please check your selections.")


if __name__ == "__main__":
    main()