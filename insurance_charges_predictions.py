# ==============================================================================
# File:         insurance_charges_prediction.py
# Author:       Dominic Scordino
# Description:  Train a regression neural network using TensorFlow to predict 
#               medical insurance charges based on demographic and lifestyle data.
# Dependencies: TensorFlow, Pandas, scikit-learn, Matplotlib
# ==============================================================================

# ------------------------------------------------------------------------------
# Import required libraries
# ------------------------------------------------------------------------------

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# ------------------------------------------------------------------------------
# Load and Inspect Data
# ------------------------------------------------------------------------------

# Load the dataset from the provided URL
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/insurance.csv")
print(insurance.head())  # Show first few rows for sanity check

# ------------------------------------------------------------------------------
# Data Preprocessing: Encoding & Normalization
# ------------------------------------------------------------------------------

# Define a column transformer to normalize numeric features and one-hot encode categorical ones
ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]),            # Normalize numeric columns
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])  # Encode categorical columns
)

# Split features (X) and target (y)
features = insurance.drop("charges", axis=1)
labels = insurance["charges"]

# Train-test split (80% train, 20% test)
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Fit the column transformer only on training data
ct.fit(features_train)

# Apply transformation to both train and test sets
features_train_normal = ct.transform(features_train)
features_test_normal = ct.transform(features_test)

# Print data shapes for verification
print("Raw Data Shape: ", features_train.shape)
print("Normalized Data Shape: ", features_train_normal.shape)

# ------------------------------------------------------------------------------
# Model Definition
# ------------------------------------------------------------------------------

# Define early stopping to prevent overfitting
callback = tf.keras.callbacks.EarlyStopping(
    monitor="mae", min_delta=10, patience=10, restore_best_weights=True
)

# Define the sequential regression model
insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu", input_shape=(features_train_normal.shape[1],), name="input"),
    tf.keras.layers.Dense(50, activation="relu"),    
	tf.keras.layers.Dense(1, name="output")  # Output a single regression value
])

# Compile the model with MAE loss and Adam optimizer
insurance_model.compile(
    loss=tf.keras.losses.MeanAbsoluteError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["mae", "mse"]
)

# ------------------------------------------------------------------------------
# Model Training
# ------------------------------------------------------------------------------

# Fit the model on the training data
history = insurance_model.fit(
    features_train_normal, 
    labels_train, 
    validation_split=0.1,
    epochs=1000, 
    callbacks=[callback], 
    verbose=1
)

insurance_model.save("insurance_model.h5")
# ------------------------------------------------------------------------------
# Evaluation and Prediction
# ------------------------------------------------------------------------------

# Evaluate the model on the test set
print("Evaluate the model...")
insurance_model.evaluate(features_test_normal, labels_test)

# Make predictions on the test set
print("Make predictions...")
insurance_model_predictions = insurance_model.predict(features_test_normal)
print(tf.squeeze(insurance_model_predictions))  # Flatten predictions for readability

# ------------------------------------------------------------------------------
# Test the Model with a Custom Input
# ------------------------------------------------------------------------------

# Define a sample input (replace with realistic values)
sample_input = pd.DataFrame({
    "age": [30], 
    "sex": ["male"], 
    "bmi": [25.0], 
    "children": [2], 
    "smoker": ["no"], 
    "region": ["southeast"]
})

# Transform the sample input using the column transformer
sample_input_normal = ct.transform(sample_input)

# Make a prediction
sample_prediction = insurance_model.predict(sample_input_normal)
print("Sample Input: \n", sample_input)
print(f"Predicted insurance charge for the sample input: ${sample_prediction[0][0]:.2f}")

# ------------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------------

# Plot training loss over epochs
pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
plt.xlabel("Epochs")
plt.ylabel("Loss (MAE)")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()

# ==============================================================================
# End of File
# ==============================================================================
