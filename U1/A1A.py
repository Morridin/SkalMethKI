#!/usr/bin/env python

import h5py
import time
import torch

class KMeans:
    def __init__(self, n_clusters=8, init="random", max_iter=300, tol=-1.0):
        self.init = init             # initialization mode (default: random)
        self.max_iter = max_iter     # maximum number of iterations
        self.n_clusters = n_clusters # number of clusters
        self.tol = tol               # tolerance for convergence criterion

        self._inertia = float("nan")
        self._cluster_centers = None

    def _initialize_centroids(self, x):
        indices = torch.randperm(x.shape[0])[: self.n_clusters]
        self._cluster_centers = x[indices]

    def _fit_to_cluster(self, x):
        distances = torch.cdist(x, self._cluster_centers)
        matching_centroids = distances.argmin(axis=1, keepdim=True)

        return matching_centroids

    def fit(self, x):
        self._initialize_centroids(x)
        new_cluster_centers = self._cluster_centers.clone()

        # Iteratively fit points to centroids.
        for idx in range(self.max_iter):
            # determine the centroids
            print("Iteration", idx, "...")
            matching_centroids = self._fit_to_cluster(x)

            # Update centroids.
            for i in range(self.n_clusters):
                # points in current cluster
                selection = (matching_centroids == i).type(torch.int64)

                # Accumulate points and total number of points in cluster.
                assigned_points = (x * selection).sum(axis=0, keepdim=True)
                points_in_cluster = selection.sum(axis=0, keepdim=True).clamp(
                    1.0, torch.iinfo(torch.int64).max
                )

                # Compute new centroids.
                new_cluster_centers[i : i + 1, :] = assigned_points / points_in_cluster

            # Check whether centroid movement has converged.
            self._inertia = ((self._cluster_centers - new_cluster_centers) ** 2).sum()
            self._cluster_centers = new_cluster_centers.clone()
            if self.tol is not None and self._inertia <= self.tol:
                break

        return self

print("PyTorch k-means clustering")

path = "/pfs/work7/workspace/scratch/ku4408-VL_ScalableAI/data/cityscapes_300.h5"
dataset = "cityscapes_data"

print("Loading data... {}[{}]".format(path, dataset), end="")
print("\n")
# Data is available in HDF5 format.
# An HDF5 file is a container for two kinds of objects:
# - datasets: array-like collections of data
# - groups: folder-like containers holding datasets and other groups
# Most fundamental thing to remember when using h5py is:
# Groups work like dictionaries, and datasets work like NumPy arrays.

# Open file for reading. We use the Cityscapes dataset.

with h5py.File(path, "r") as handle:
    print("Open h5 file...")
    data = torch.tensor(handle[dataset][:300], device="cpu") # default: device ="cpu"; set device="cuda" for GPU
print("Torch tensor created...")

# k-means parameters
num_clusters = 8
num_iterations = 20

kmeans = KMeans(n_clusters=num_clusters, max_iter=num_iterations)
print("Start fitting the data...")
start = time.perf_counter() # Start runtime measurement.
kmeans.fit(data)            # Perform actual k-means clustering.
end = time.perf_counter()   # Stop runtime measurement.
print("DONE.")
print("Run time:","\t{}s".format(end - start), "s")