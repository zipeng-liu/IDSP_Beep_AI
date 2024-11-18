from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np
from pyproj import Transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Step 1: Data Processing
def process_crime_data(input_file='data/crime_data.csv', output_file='data/processed_crime_data.csv'):
    print("Processing crime data...")
    current_crs = "epsg:26910"  # Example projection; adjust as needed
    target_crs = "epsg:4326"    # WGS84 (latitude/longitude)

    transformer = Transformer.from_crs(current_crs, target_crs, always_xy=True)

    # Load the CSV data
    data = pd.read_csv(input_file)

    # Convert X, Y to latitude and longitude
    def convert_to_latlon(x, y):
        lon, lat = transformer.transform(x, y)
        return lat, lon

    data[['latitude', 'longitude']] = data.apply(lambda row: pd.Series(convert_to_latlon(row['X'], row['Y'])), axis=1)
    data['crime_count'] = 1  # Each row represents an incident

    # Group by latitude and longitude to calculate crime density
    crime_density = data.groupby(['latitude', 'longitude']).size().reset_index(name='crime_count')

    # Normalize crime counts
    scaler = MinMaxScaler()
    crime_density['normalized_crime_count'] = scaler.fit_transform(crime_density[['crime_count']])

    # Save the processed data
    crime_density.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    return crime_density

# Step 2: Model Training
def train_model(data, model_path='models/safety_score_model.h5'):
    print("Training model...")
    X = data[['latitude', 'longitude']].values
    y = data['normalized_crime_count'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define and compile the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

    # Save the model
    model.save(model_path)
    print(f"Model training complete and saved to {model_path}")
    return model

# Check if the processed data and model already exist
if not os.path.exists('data/processed_crime_data.csv'):
    processed_data = process_crime_data()
else:
    processed_data = pd.read_csv('data/processed_crime_data.csv')

if not os.path.exists('models/safety_score_model.h5'):
    model = train_model(processed_data)
else:
    print("Loading existing model...")
    model = tf.keras.models.load_model('models/safety_score_model.h5')

# Step 3: Flask API
app = Flask(__name__)

@app.route('/getSafeRoute', methods=['GET'])
def get_safe_route():
    start = request.args.get('start')  # e.g., "49.2827,-123.1207"
    end = request.args.get('end')      # e.g., "49.2876,-123.1181"

    # Validate input
    if not start or not end:
        return jsonify({"error": "Missing 'start' or 'end' query parameter"}), 400

    try:
        start_coords = [float(x) for x in start.split(',')]
        end_coords = [float(x) for x in end.split(',')]

        # Generate route with interpolation and safety score prediction
        route = [start_coords]
        steps = 10  # Number of interpolation points

        for i in range(1, steps):
            lat = start_coords[0] + (end_coords[0] - start_coords[0]) * i / steps
            lon = start_coords[1] + (end_coords[1] - start_coords[1]) * i / steps
            safety_score = model.predict(np.array([[lat, lon]]))[0][0]

            # Only add to the route if the safety score is below a certain threshold
            if safety_score < 0.5:  # Adjust threshold as needed
                route.append([lat, lon])

        route.append(end_coords)
        return jsonify({"route": route})

    except ValueError as ve:
        return jsonify({"error": f"Invalid format for 'start' or 'end' coordinates: {ve}"}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
