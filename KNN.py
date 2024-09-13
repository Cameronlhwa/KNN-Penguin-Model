# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as pltgraph
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set a random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Function to calculate Euclidean distance between two points
def compute_euclidean_dist(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# K-Nearest Neighbors (KNN) algorithm class
class KNNClassifier:

    def __init__(self, num_neighbors=3):
        self.num_neighbors = num_neighbors

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self

    def predict(self, X_test):
        return np.array([self._make_prediction(x) for x in X_test])

    def _make_prediction(self, single_point):
        # Calculate distances from the test point to all training points
        distances = [compute_euclidean_dist(single_point, x_train) for x_train in self.X_train]

        # Sort distances and select the nearest neighbors
        nearest_neighbor_indices = np.argsort(distances)[:self.num_neighbors]

        # Retrieve the labels of the nearest neighbors
        nearest_neighbor_labels = self.y_train[nearest_neighbor_indices]

        # Return the most frequent label among the nearest neighbors
        label_counts = np.bincount(nearest_neighbor_labels)
        return np.argmax(label_counts)

# Load the penguins dataset and clean it
dataset = sns.load_dataset('penguins').dropna()

# Visualize the dataset with a scatter plot
sns.relplot(x='flipper_length_mm', y='bill_length_mm', hue='species', data=dataset)

# Convert categorical columns to numerical ones using one-hot encoding
dataset = pd.get_dummies(dataset, columns=['island', 'sex'])

# Encode the species column as numerical values
dataset['species'], _ = dataset['species'].factorize()

# Preview the first few rows of the dataset
dataset.head()

# Separate the dataset into features (X) and target variable (y)
X = dataset.drop('species', axis=1).to_numpy()
y = dataset['species'].to_numpy()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Standardize the features using a scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the custom KNN classifier
knn_classifier = KNNClassifier(num_neighbors=3).fit(X_train, y_train)

# Make predictions using the trained KNN model
y_predictions = knn_classifier.predict(X_test)

# Display the accuracy of the custom KNN classifier
print(f"Custom KNN Model Accuracy: {accuracy_score(y_test, y_predictions):.3f}")

# Use the sklearn built-in KNeighborsClassifier for comparison
from sklearn.neighbors import KNeighborsClassifier

# Train the sklearn KNN model
sklearn_knn_model = KNeighborsClassifier(n_neighbors=9).fit(X_train, y_train)

# Make predictions using the sklearn KNN model
y_sklearn_predictions = sklearn_knn_model.predict(X_test)

# Display the accuracy of the sklearn KNN classifier
print(f"Sklearn KNN Model Accuracy: {accuracy_score(y_test, y_sklearn_predictions):.3f}")

# Get the first two feature columns for visualization (e.g., two feature dimensions for 2D plotting)
X_test_plot = X_test[:, :2]  # Select the first two columns of the test data

# Plot the actual test labels in the test set
pltgraph.figure(figsize=(10, 6))
pltgraph.scatter(X_test_plot[:, 0], X_test_plot[:, 1], c=y_test, marker='o', label='Actual', cmap='coolwarm', edgecolor='k')

# Plot the KNN predicted labels for the same data points
pltgraph.scatter(X_test_plot[:, 0], X_test_plot[:, 1], c=y_predictions, marker='x', label='Predicted', cmap='coolwarm')

# Adding labels and title
pltgraph.title('KNN Predictions vs Actual Labels')
pltgraph.xlabel('Feature 1 (e.g., flipper length)')
pltgraph.ylabel('Feature 2 (e.g., bill length)')
pltgraph.legend(['Actual', 'Predicted'])

# Show the plot
pltgraph.show()