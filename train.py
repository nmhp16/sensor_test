# train.py
import glob
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib # For saving the trained model
import warnings

# Suppress specific warnings if needed
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

dataset_dir = "dataset" # Directory where collect.py saves the .npy files
model_filename = "sonar_classifier.joblib" # Filename to save the trained model

def extract_features(distances):
    """
    Extracts features from a list or array of sonar distances.
    Args:
        distances (list or np.ndarray): List/array of detected echo distances.
    Returns:
        list: A list of float features. Returns None if input is invalid.
    """
    # Ensure input is iterable and convert to numpy array
    try:
        if distances is None:
             distances = np.array([])
        elif not isinstance(distances, np.ndarray):
             distances = np.array(distances, dtype=np.float32)
    except Exception as e:
        print(f"Warning: Could not convert distances to array: {e}. Input: {distances}")
        return None # Indicate error

    # Ensure data is numeric and finite before processing
    distances = distances[np.isfinite(distances)]

    if distances.size == 0:
        # Define features for empty or invalid sonar data
        return [0.0, 0.0, 0.0, 0.0, 0.0] # Use floats
    else:
        distances = np.sort(distances) # Sort valid distances
        diffs = np.diff(distances) # Calculate differences
        features = [
            float(len(distances)),            # Number of detected distances
            float(np.min(distances)),         # Minimum distance
            float(np.max(distances)),         # Maximum distance
            float(np.mean(distances)),        # Mean distance
            float(np.std(diffs)) if len(diffs) > 0 else 0.0 # Std dev of differences
        ]
        return features


def load_dataset():
    """Loads the dataset from label-specific subdirectories."""
    x, y = [], []
    print(f"Loading data from subdirectories in: {os.path.abspath(dataset_dir)}")

    # Check if dataset directory exists
    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found.")
        return np.array([]), np.array([])

    skipped_files = 0
    processed_files = 0
    # Iterate through items in the dataset directory
    for label in os.listdir(dataset_dir):
        label_dir = os.path.join(dataset_dir, label)
        # Check if it's a directory (representing a class label)
        if os.path.isdir(label_dir):
            print(f"Processing label: {label}")
            # Find all .npy files within the label directory
            files_found = glob.glob(os.path.join(label_dir, "*.npy"))
            if not files_found:
                print(f"  Warning: No .npy files found for label '{label}'.")
                continue

            for file in files_found:
                filename = os.path.basename(file)
                try:
                    # Load the numpy array of distances
                    data = np.load(file, allow_pickle=True)

                    # Basic check if data looks like a list/array of numbers
                    if not isinstance(data, np.ndarray) or data.ndim != 1:
                         if isinstance(data, list): data = np.array(data, dtype=np.float32)
                         else:
                             print(f"  Warning: Skipping file - expected 1D array/list, got {type(data)} in {filename}")
                             skipped_files += 1
                             continue

                    features = extract_features(data) # Pass the loaded distances array

                    if features is None:
                        print(f"  Warning: Skipping file due to feature extraction error: {filename}")
                        skipped_files += 1
                        continue

                    x.append(features)
                    y.append(label) # Use the directory name as the label
                    processed_files += 1

                except Exception as e:
                    print(f"  Error loading or processing file {filename}: {e}")
                    skipped_files += 1
        else:
             print(f"Skipping non-directory item: {label}")


    print(f"\nSuccessfully processed {processed_files} files, skipped {skipped_files} files.")
    if not x: # Check if list is empty after processing files
         print("Error: No valid data loaded from files.")
         return np.array([]), np.array([])

    # Ensure features are floats for model training consistency
    return np.array(x, dtype=np.float32), np.array(y)


def train_model():
    """Loads data, trains the model, evaluates, and saves it."""
    X, y = load_dataset()

    if X.size == 0 or y.size == 0: # Check if arrays are empty
        print("No data loaded. Cannot train model.")
        return None

    # Check number of samples
    num_samples = X.shape[0]
    if num_samples == 0:
        print("No valid samples loaded. Cannot train model.")
        return None

    # Check number of features dynamically
    if X.ndim != 2:
        print(f"Error: Feature array has unexpected dimensions: {X.ndim}")
        return None
    num_features = X.shape[1]
    if num_features == 0:
        print("Error: Loaded data has 0 features.")
        return None

    print(f"\nTraining with {num_samples} samples and {num_features} features.")
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"Classes found: {dict(zip(unique_labels, counts))}")


    # Split data only if enough samples and classes exist for stratification
    test_size = 0.2
    min_samples_per_class_train = 1 # Minimum needed by RandomForest
    min_samples_per_class_test = 1 # Minimum needed for meaningful evaluation
    min_classes_for_stratify = 2

    # Check if stratification is possible and meaningful
    can_stratify = len(unique_labels) >= min_classes_for_stratify and \
                   all(counts >= min_samples_per_class_train + min_samples_per_class_test)

    if num_samples >= 5 and can_stratify: # Need enough samples overall and per class
         X_train, X_test, y_train, y_test = train_test_split(
             X, y,
             test_size=test_size,
             random_state=42,
             stratify=y # Stratify helps keep class proportions in train/test
         )
         print(f"Split data: {len(X_train)} train, {len(X_test)} test samples.")
    else:
        print("Warning: Not enough data or classes for a stratified train/test split. Training on all data.")
        X_train, y_train = X, y
        # Create empty test set if split wasn't possible
        X_test, y_test = np.array([], dtype=np.float32).reshape(0, num_features), np.array([], dtype=y.dtype)


    # --- Train the RandomForestClassifier ---
    # Use class_weight='balanced' to help with imbalanced datasets
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    try:
        clf.fit(X_train, y_train)
        print("Model training complete.")
    except Exception as e:
        print(f"Error during model training: {e}")
        return None


    # --- Evaluate ---
    if X_test.size > 0 and y_test.size > 0:
        try:
            y_pred = clf.predict(X_test)
            print("\nClassification Report (Test Set):")
            # Ensure labels for report match those present in y_test and y_pred
            report_labels = np.unique(np.concatenate((y_test, y_pred)))
            print(classification_report(y_test, y_pred, labels=report_labels, zero_division=0))
        except Exception as e:
            print(f"Could not generate classification report: {e}")
    else:
        print("\nNo test set available for evaluation.")

    # --- Save the model ---
    try:
        joblib.dump(clf, model_filename)
        print(f"\nSaved trained model to: {model_filename}")
    except Exception as e:
        print(f"Error saving model with joblib: {e}")

    return clf

# --- Main execution block ---
if __name__ == "__main__":
    print("--- Starting Model Training ---")
    trained_classifier = train_model()
    if trained_classifier:
        print("\n--- Training Script Finished Successfully ---")
    else:
        print("\n--- Training Script Finished With Errors or No Data ---")
