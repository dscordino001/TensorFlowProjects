# import the required libraries
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# borrowed classes from sci-kit learning to prepare data
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Read in the insurance dataset
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/insurance.csv")
print(insurance)

# Create a column transformer to pass columns into NN
ct = make_column_transformer(
	# turn all integer values in these columns between 0 and 1.
	(MinMaxScaler(), ["age", "bmi", "children"]),
	# if there are any columns the oneHotEncoder doesn't know about, just ignore them
	(OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)

# Create our X and Y values (features and labels)
features = insurance.drop("charges", axis=1)
labels = insurance["charges"]

# Build train and test data sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Fit the column transformer to our training data
# When you have a column transformer, you want to fit it to your training data, then use that fit column transformer to fit your test data
ct.fit(features_train)

# Transform training and test data with normalization (OneHotEncoder and MinMaxScaler)
features_train_normal = ct.transform(features_train)
features_test_normal = ct.transform(features_test)

# Visualize Data and Check Data Shapes
print(features_train_normal)
print("Raw Data Shape: ", features_train.shape)
print("Normalized Data Shape: ", features_train_normal.shape)

# Create Early Stopping Callback: This will stop the model if it stops improving by 3 MAE over 10 epochs
callback = tf.keras.callbacks.EarlyStopping(monitor="mae", min_delta=3, patience=10)

# Build NN model to fit normalized data
insurance_model = tf.keras.Sequential([
	tf.keras.layers.Dense(100, name="input"),
	tf.keras.layers.Dense(10),
	tf.keras.layers.Dense(1, name="output")
])

# Model compile
insurance_model.compile(
	loss=tf.keras.losses.mae,
	optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
	metrics=["mae"]
)

history = insurance_model.fit(features_train_normal, labels_train, epochs=1000000, callbacks=[callback])\

# Evaluate the model
print("Evaluate the model...")
insurance_model.evaluate(features_test_normal, labels_test)

# Make predictions
print("Make predictions...")
insurance_model_predictions = insurance_model.predict(features_test_normal)
print(tf.squeeze(insurance_model_predictions))

# print dataFrame of loss curve
pd.DataFrame(history.history).plot()
plt.xlabel("Loss")
plt.ylabel("Epochs")
plt.title("Loss Curve")
plt.show()