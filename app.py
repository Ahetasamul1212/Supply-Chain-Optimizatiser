import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import streamlit as st

# Load the dataset
data = pd.read_csv("Supply chain logisitcs problem.csv")

# Preprocess the data
# Drop irrelevant columns
data_cleaned = data.drop(columns=["Order ID", "Order Date", "Customer", "Product ID"])

# Convert all columns to float where possible
for col in data_cleaned.columns:
    try:
        data_cleaned[col] = data_cleaned[col].astype(float)
    except ValueError:
        pass

# Encode categorical columns
categorical_columns = ["Origin Port", "Carrier", "Service Level", "Plant Code", "Destination Port"]
label_encoders = {col: LabelEncoder() for col in categorical_columns}
for col in categorical_columns:
    data_cleaned[col] = label_encoders[col].fit_transform(data_cleaned[col])

# Separate features and target variable
X = data_cleaned.drop(columns=["Ship Late Day count"])
y = data_cleaned["Ship Late Day count"]

# Normalize numerical columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Model RMSE: {rmse:.2f}")

# Streamlit App
st.title("Ship Late Day Predictor")

# Input fields
fields = [
    ("Origin Port (encoded)", "float"),
    ("Carrier (encoded)", "float"),
    ("Service Level (encoded)", "float"),
    ("Ship Ahead Day Count", "float"),
    ("Plant Code (encoded)", "float"),
    ("Destination Port (encoded)", "float"),
    ("Unit Quantity", "float"),
    ("Weight", "float"),
    ("TPT", "float"),
]

inputs = {}
for label_text, input_type in fields:
    if input_type == "float":
        inputs[label_text] = st.number_input(label_text)
    else:
        inputs[label_text] = st.text_input(label_text)

# Predict button
if st.button("Predict"):
    try:
        # Get inputs from the user
        inputs_list = list(inputs.values())

        # Scale inputs
        inputs_scaled = scaler.transform([inputs_list])

        # Predict
        prediction = model.predict(inputs_scaled)[0]
        st.write(f"Predicted Ship Late Day count: {prediction:.2f}")
    except Exception as e:
        st.error(f"Invalid input: {str(e)}")