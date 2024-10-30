import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
file_path = r"C:\Users\esthe\OneDrive\Documents\Machine Learning\bezdekIris.data"  # Using raw string
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_data = pd.read_csv(file_path, header=None, names=column_names)

# Drop the 'class' column since we don't need it for clustering
X = iris_data.drop(columns=['class'])

# Step 1: Initial Data Visualization
plt.scatter(X['sepal_length'], X['sepal_width'], s=50)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Initial Data Distribution')
plt.savefig("initial_data_distribution_kmeans.png")  # Save the initial plot
plt.show()

# Step 2: K-means Clustering
kmeans = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot K-means final clusters and save
plt.scatter(X['sepal_length'], X['sepal_width'], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title("K-means Clustering (Final Clusters)")
plt.legend()
plt.savefig("kmeans_final_clusters.png")  # Save the K-means cluster plot
plt.show()

# Print epoch size and error rate (inertia)
print("K-means Epoch Size (Number of iterations):", kmeans.n_iter_)
print("K-means Inertia (Error Rate):", kmeans.inertia_)
