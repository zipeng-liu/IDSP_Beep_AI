import pandas as pd
from pyproj import Transformer
from sklearn.preprocessing import MinMaxScaler

# Replace 'epsg:XXXX' with the appropriate EPSG code for your data's current projection
current_crs = "epsg:26910"  # Example: NAD83 / UTM zone 10N (adjust as needed)
target_crs = "epsg:4326"    # WGS84 (latitude/longitude)

# Initialize the transformer
transformer = Transformer.from_crs(current_crs, target_crs, always_xy=True)

# Load the CSV data
data = pd.read_csv('data/crime_data.csv')

# Convert X, Y coordinates to latitude and longitude
def convert_to_latlon(x, y):
    lon, lat = transformer.transform(x, y)  # Transform returns (lon, lat)
    return lat, lon

# Apply conversion to each row and store latitude and longitude in new columns
data[['latitude', 'longitude']] = data.apply(lambda row: pd.Series(convert_to_latlon(row['X'], row['Y'])), axis=1)

# Add a column to count incidents in grid areas (assuming each row is an incident)
data['crime_count'] = 1

# Group by latitude and longitude to calculate crime density
crime_density = data.groupby(['latitude', 'longitude']).size().reset_index(name='crime_count')

# Normalize crime counts using MinMaxScaler
scaler = MinMaxScaler()
crime_density['normalized_crime_count'] = scaler.fit_transform(crime_density[['crime_count']])

# Save the processed data for model training
crime_density.to_csv('data/processed_crime_data.csv', index=False)

print("Data processing and conversion complete. Saved to 'data/processed_crime_data.csv'.")
