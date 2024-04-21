import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
# This is a placeholder path. Replace it with the path to your actual dataset file.
# Your dataset should be in a CSV format with features and a decade label for each song.
dataset_path = 'song_features_dataset.csv'
data = pd.read_csv(dataset_path)

# Assuming your dataset has columns for features like 'feature1', 'feature2', ..., 'featureN' and 'decade' for the label
X = data[['feature1', 'feature2', 'featureN']]  # Replace these with your actual feature columns
y = data['decade']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))
