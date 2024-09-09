import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import gaussian_kde

# Init lists for data
words = []
embeddings = []
counts = []

# Read the embeddings.tsv file
with open("output/embeddings.tsv", "r") as file:
    for line in file:
        # Split the line at the last comma
        parts = line.split("\t")
        if len(parts) == 3:
            word = parts[0].strip()
            count = int(parts[1].strip())
            embedding_str = parts[2].strip()

            # Convert the embedding string to a list of floats
            embedding = np.array([float(x) for x in embedding_str.split()])

            words.append(word)
            embeddings.append(embedding)
            counts.append(count)


# Find most common words
all_words = [word for phrase in words for word in phrase.split()]
word_counts = Counter(all_words)
most_common_words = word_counts.most_common()
print("Most common words:", most_common_words)

# Convert lists to numpy arrays
words = np.array(words)
embeddings = np.array(embeddings)
counts = np.array(counts)

# Apply PCA
pca = PCA(n_components=2, random_state=42)
embeddings_pca = pca.fit_transform(embeddings)

# Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(embeddings_pca)

# Apply KMeans
kmeans = KMeans(n_clusters=9, random_state=42)
kmeans.fit(embedding_2d)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plotting
fig, ax = plt.subplots(figsize=(16, 12))

# Voronoi overlay
vor = Voronoi(centroids)
voronoi_plot_2d(
    vor,
    ax=ax,
    show_vertices=False,
    line_colors="gray",
    line_width=2,
    line_alpha=0.75,
    point_size=0,
)

# Point sizes relative to counts
min_size, max_size = 2, 100
min_count, max_count = min(counts), max(counts)
sizes = [
    (max_size - min_size) * (count - min_count) / (max_count - min_count) + min_size
    for count in counts
]

# Scatter plot colored by cluster
sc = ax.scatter(
    embedding_2d[:, 0],
    embedding_2d[:, 1],
    c=labels,
    alpha=0.6,
    s=sizes,
    cmap="Spectral",
)

# Randomly sample points to label
num_labels = len(words) // 4
np.random.seed(42)  # for reproducibility
sample_indices = np.random.choice(len(words), num_labels, replace=False)

# Add labels for the sampled points
for idx in sample_indices:
    plt.annotate(
        words[idx],
        (embedding_2d[idx, 0], embedding_2d[idx, 1]),
        xytext=(5, 2),
        textcoords="offset points",
        fontsize=8,
        alpha=0.7,
    )

# Do the plot
plt.tight_layout()
plt.grid(True)
plt.savefig("output/embeddings.pdf", bbox_inches="tight", dpi=300)
plt.close()


# Contour plot
def plot_smooth_contours(X, labels, ax=None, bandwidth_factor=0.15):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    # Calculate the point density for all points with reduced bandwidth
    xy = np.vstack([X[:, 0], X[:, 1]])
    kde = gaussian_kde(xy, bw_method=bandwidth_factor)

    # Create a grid of points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # Calculate the kernel density estimation
    z = np.reshape(kde(positions).T, xx.shape)

    # Plot filled contours
    contour = ax.contourf(xx, yy, z, levels=20, cmap="viridis", alpha=0.6)

    # Plot contour lines
    ax.contour(xx, yy, z, levels=20, colors="k", linewidths=0.5, alpha=0.3)

    # Plot the points colored by cluster
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax.scatter(X[mask, 0], X[mask, 1], c=[color], alpha=0.6, s=20)

    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_title("Cluster Visualization with Less Smooth Contours")

    # Add a colorbar
    plt.colorbar(contour, ax=ax, label="Density")

    return ax


fig, ax = plt.subplots(figsize=(16, 12))

# Plot smooth contours
plot_smooth_contours(embedding_2d, labels, ax=ax)

# Add labels for sampled points
num_labels = len(words) // 4
np.random.seed(42)
sample_indices = np.random.choice(len(words), num_labels, replace=False)
for idx in sample_indices:
    ax.annotate(
        words[idx],
        (embedding_2d[idx, 0], embedding_2d[idx, 1]),
        xytext=(5, 2),
        textcoords="offset points",
        fontsize=8,
        alpha=0.7,
    )

plt.legend()
plt.tight_layout()
plt.savefig("smooth_cluster_contours_with_labels.png", dpi=300, bbox_inches="tight")
plt.close()
