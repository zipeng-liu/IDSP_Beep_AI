# README

## Overview
This project helps users find safer routes by analyzing crime data. It consists of three main parts:

1. **Preparing Data**.
2. **Training an AI Model**.
3. **Providing Safe Routes Using a Web App**.

## Step-by-Step Guide

### 1. Preparing the Data (`preprocess_data.py`)
This script processes raw crime data for model training.

**Steps**:
- **Read Data**: Reads crime data from a CSV file containing locations of incidents.
- **Convert Locations**: Changes map coordinates from a projected format to latitude and longitude (real-world map locations).
- **Count Crimes**: Counts the number of incidents at each location.
- **Normalize Data**: Scales the crime counts between 0 and 1 for easier model training.
- **Save Data**: Saves the processed data to a new CSV file for use in the training step.

### 2. Training an AI Model (`train_model.py`)
This script trains a neural network model to predict safety scores based on the processed crime data.

**Steps**:
- **Read Processed Data**: Loads the data prepared by `preprocess_data.py`.
- **Select Features**: Uses latitude and longitude as inputs and the normalized crime count as the output.
- **Split Data**: Splits the data into training and testing sets.
- **Create the Model**: Builds an AI model with TensorFlow, consisting of three layers to learn patterns in the data.
- **Train the Model**: Trains the model to predict safety scores based on the given locations.
- **Save the Model**: Saves the trained model to a file for future use.

### 3. Providing Safe Routes (`app.py`)
This script sets up a web app using Flask, allowing users to request safer routes between two locations.

**Steps**:
- **Set Up the Web App**: Uses Flask to create a web service.
- **Load Data and Model**: Checks if the processed data and trained model already exist. If not, runs the necessary scripts to create them.
- **Handle Route Requests**: When a user provides start and end locations:
  - The app calculates intermediate points along the route.
  - It uses the trained AI model to predict safety scores for each point.
  - Points with low crime scores are added to the route to ensure safety.
- **Return the Route**: Responds with a route containing safe points as a JSON object.

## How to Run the Project
1. **Data Preparation**: Run `preprocess_data.py` to process the raw data.
2. **Model Training**: Run `train_model.py` to train and save the AI model.
3. **Start the Web App**: Run `app.py` to start the Flask server. Use the `/getSafeRoute` endpoint to request safe routes by providing start and end coordinates.

## Project Requirements
- Python libraries: `Flask`, `TensorFlow`, `Pandas`, `Numpy`, `pyproj`, `scikit-learn`
- Data file: `data/crime_data.csv`
- Directory structure:
  - `data/`: Contains raw and processed CSV files.
  - `models/`: Contains the trained AI model file.

## Summary
This project processes crime data, trains an AI model to predict safety scores, and provides a web service to suggest safer routes based on user-provided locations.
