import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load inertial signals dynamically
def load_inertial_signals(data_type, signal_types, data_path):
    """
    Load inertial signal data for the specified type (train or test).
    Args:
        data_type (str): Either 'train' or 'test'.
        signal_types (list): List of signal prefixes, e.g., ['total_acc', 'body_acc', 'body_gyro'].
        data_path (str): Path to the dataset folder.
    Returns:
        dict: Dictionary where keys are signal names and values are signal data arrays.
    """
    inertial_data = {}
    for signal in signal_types:
        for axis in ['x', 'y', 'z']:
            file_name = f"{signal}_{axis}_{data_type}.txt"
            file_path = os.path.join(data_path, data_type, "Inertial Signals", file_name)
            inertial_data[f"{signal}_{axis}"] = np.loadtxt(file_path)
    return inertial_data

# Function to combine X, Y, Z axes for each signal type
def combine_axes(signal_dict, signal_types):
    combined = []
    for signal in signal_types:
        combined.append(np.stack([signal_dict[f"{signal}_x"], 
                                  signal_dict[f"{signal}_y"], 
                                  signal_dict[f"{signal}_z"]], axis=-1))
    return combined

# Function to extract features (mean, std) for each signal type
def extract_signal_features(combined_signals):
    features = []
    for signal in combined_signals:
        mean = np.mean(signal, axis=1)  # Mean across time steps
        std = np.std(signal, axis=1)    # Std deviation across time steps
        features.append(np.hstack([mean, std]))
    return np.hstack(features)

# Paths and signal types
data_path = "UCI HAR Dataset"  # Adjust this to the dataset root folder if needed
signal_types = ["total_acc", "body_acc", "body_gyro"]

# Load inertial signals
train_inertial_signals = load_inertial_signals("train", signal_types, data_path)
test_inertial_signals = load_inertial_signals("test", signal_types, data_path)

# Combine axes
train_combined_signals = combine_axes(train_inertial_signals, signal_types)
test_combined_signals = combine_axes(test_inertial_signals, signal_types)

# Extract features
train_signal_features = extract_signal_features(train_combined_signals)
test_signal_features = extract_signal_features(test_combined_signals)

# Load precomputed features
# Updated reading of the files using sep='\s+'
X_train = pd.read_csv(os.path.join(data_path, "train", "X_train.txt"), sep='\s+', header=None).values
X_test = pd.read_csv(os.path.join(data_path, "test", "X_test.txt"), sep='\s+', header=None).values

# Normalize precomputed features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Combine precomputed and engineered features
X_train_combined = np.hstack([X_train_scaled, train_signal_features])
X_test_combined = np.hstack([X_test_scaled, test_signal_features])

# Load activity labels
y_train = pd.read_csv(os.path.join(data_path, "train", "y_train.txt"), header=None).squeeze()
y_test = pd.read_csv(os.path.join(data_path, "test", "y_test.txt"), header=None).squeeze()

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_combined, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test_combined)
print(y_pred)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
plt.savefig("confusion_matrix_plot.png", format="png")  # Save as PNG file
