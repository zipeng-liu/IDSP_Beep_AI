import json
import tensorflow as tf
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

# Load the trained model
model_path = 'models/safety_score_model.h5'
model = tf.keras.models.load_model(model_path)
logging.info(f"Model loaded from {model_path}")

def calculate_safe_route(start_coords, end_coords):
    """Calculate a safe route between two points."""
    route = [start_coords]
    steps = 10

    for i in range(1, steps):
        lat = start_coords[0] + (end_coords[0] - start_coords[0]) * i / steps
        lon = start_coords[1] + (end_coords[1] - start_coords[1]) * i / steps
        safety_score = model.predict(np.array([[lat, lon]]))[0][0]
        logging.info(f"Step {i}: ({lat}, {lon}) - Safety score: {safety_score}")

        if safety_score < 0.5:  # Threshold for safety
            route.append([lat, lon])

    route.append(end_coords)
    return route

def lambda_handler(event, context):
    """AWS Lambda entry point."""
    try:
        logging.info(f"Received event: {event}")

        # Extract query parameters
        query_params = event.get('queryStringParameters', {})
        start = query_params.get('start')
        end = query_params.get('end')

        if not start or not end:
            return {
                'statusCode': 400,
                'body': json.dumps({"error": "Missing 'start' or 'end' query parameter"})
            }

        try:
            start_coords = [float(x) for x in start.split(',')]
            end_coords = [float(x) for x in end.split(',')]

            # Calculate the safe route
            route = calculate_safe_route(start_coords, end_coords)

            return {
                'statusCode': 200,
                'body': json.dumps({"route": route}),
                'headers': {'Content-Type': 'application/json'}
            }

        except ValueError as ve:
            logging.error(f"Invalid input format: {ve}")
            return {
                'statusCode': 400,
                'body': json.dumps({"error": f"Invalid format for 'start' or 'end' coordinates: {ve}"})
            }

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({"error": "Internal server error"}),
            'headers': {'Content-Type': 'application/json'}
        }

# Local Testing (Optional)
if __name__ == '__main__':
    # Simulate AWS Lambda locally
    sample_event = {
        "queryStringParameters": {
            "start": "49.2827,-123.1207",
            "end": "49.2870,-123.1121"
        }
    }
    print(lambda_handler(sample_event, None))
