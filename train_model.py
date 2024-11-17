import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load processed data
data = pd.read_csv('data/processed_crime_data.csv')

# Use latitude and longitude as features
X = data[['latitude', 'longitude']].values
y = data['normalized_crime_count'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),  # Input layer with latitude and longitude
    tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer
    tf.keras.layers.Dense(1)  # Output layer for predicting safety score
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Save the model
model.save('models/safety_score_model.h5')

print("Model training complete and saved.")
