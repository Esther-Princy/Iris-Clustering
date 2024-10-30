import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

# Load the dataset
file_path = r"C:\Users\esthe\OneDrive\Documents\Machine Learning\bezdekIris.data"  # Update this path
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_data = pd.read_csv(file_path, header=None, names=column_names)

# Drop the 'class' column since we don't need it for clustering
X = iris_data.drop(columns=['class'])

# Step 1: Initial Data Visualization
plt.scatter(X['sepal_length'], X['sepal_width'], s=50)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Initial Data Distribution')
plt.savefig("initial_data_distribution_agg.png")  # Save the initial plot
plt.show()

# Step 2: Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
y_agg = agg_clustering.fit_predict(X)

# Calculate a "pseudo" inertia for demonstration purposes
# This calculates the average distance of points to their cluster center
centers = []
for cluster in range(3):  # Assuming 3 clusters
    cluster_points = X[y_agg == cluster]
    center = cluster_points.mean(axis=0)  # Calculate cluster center
    centers.append(center)

centers = pd.DataFrame(centers, columns=X.columns)
inertia = sum(pairwise_distances(X, centers).min(axis=1))

# Plot Agglomerative Clustering final clusters and save
plt.scatter(X['sepal_length'], X['sepal_width'], c=y_agg, s=50, cmap='viridis')
plt.scatter(centers['sepal_length'], centers['sepal_width'], s=200, c='red', marker='X', label='Centroids')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title("Agglomerative Clustering (Final Clusters)")
plt.legend()
plt.savefig("agg_final_clusters.png")  # Save the Agglomerative clustering plot
plt.show()

# Print results
print("Agglomerative Clustering completed.")
print("Number of clusters formed:", len(set(y_agg)))
print("Pseudo Inertia (Error Rate):", inertia)
