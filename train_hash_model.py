import pandas as pd import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# Load the dataset
df = pd.read_csv('balanced_train.csv')

# Ensure correct column names
X = df.iloc[:, 1] # Second column is md5hash y = df.iloc[:, -1] # Last column is benign

# Encode hash strings to numbers encoder = LabelEncoder()
X_encoded = encoder.fit_transform(X)


# Train the model
model = RandomForestClassifier() model.fit(X_encoded.reshape(-1, 1), y)

# Save the model and encoder joblib.dump(model, 'hash_model.joblib') joblib.dump(encoder, 'hash_encoder.joblib')

print("✅ hash_model.joblib and hash_encoder.joblib saved successfully.")
