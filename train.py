# train.py
import glob
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

dataset_dir = "dataset"

def extract_features(distances):
    # Extract features from the distances
    if not distances:
        return [0, 0, 0, 0, 0] 
    
    distances = sorted(distances) # Sort distances for consistent feature extraction
    diffs = np.diff(distances) # Calculate differences between consecutive distances
    return [
        len(distances),  # Number of detected distances
        np.min(distances), # Minimum distance
        np.max(distances), # Maximum distance
        np.mean(distances), # Mean distance
        np.std(diffs) if len(diffs) > 0 else 0 # Standard deviation of differences
    ]

def load_dataset():
    x, y = [], []
    for file in glob.glob(os.path.join(dataset_dir, "*.npy")):
        filename = os.path.basename(file) # Extract label from filename
        label = filename.split("_")[0] # Assuming label is the prefix before the first underscore
        data = np.load(file, allow_pickle=True) # Load the distances
        features = extract_features(data)   # Extract features from distances
        x.append(features) # Append features to x
        y.append(label) # Append label to y
    return np.array(x), np.array(y)

def train_model():
    X, y = load_dataset()
    if len(X) == 0:
        print("No data found.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return clf

if __name__ == "__main__":
    train_model()
