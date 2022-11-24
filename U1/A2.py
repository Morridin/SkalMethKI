#!/usr/bin/env python

import h5py
import time
import torch
from mpi4py import MPI

class KMeans:
    def __init__(self, n_clusters=8, init="random", max_iter=300, tol=-1.0):
        self.init = init             # initialization mode (default: random)
        self.max_iter = max_iter     # maximum number of iterations
        self.n_clusters = n_clusters # number of clusters
        self.tol = tol               # tolerance for convergence criterion
        
        self._inertia = float("nan")
        self._cluster_centers = None

    def _initialize_centroids(self, x):
        if rank == 0:
            indices = torch.randperm(x.shape[0])[: self.n_clusters]
            self._cluster_centers = x[indices]
        self._cluster_centers = MPI.COMM_WORLD.bcast([self._cluster_centers, MPI.FLOAT])

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
                    0.0, torch.iinfo(torch.int64).max)
                
                a_p_global = torch.empty(assigned_points.shape, dtype=torch.float)
                p_i_c_global = torch.empty(points_in_cluster.shape, dtype=torch.int64)
                
                # Communication
                MPI.COMM_WORLD.Allreduce(assigned_points, a_p_global)
                MPI.COMM_WORLD.Allreduce(points_in_cluster p_i_c_global)
                
                # Compute new centroids.
                new_cluster_centers[i : i + 1, :] = a_p_global / p_i_c_global.clamp(
                    1.0, torch.iinfo(torch.int64).max)

            # Check whether centroid movement has converged.
            self._inertia = ((self._cluster_centers - new_cluster_centers) ** 2).sum()
            self._cluster_centers = new_cluster_centers.clone()
            if self.tol is not None and self._inertia <= self.tol:
                break

        return self

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

if rank==0: print("PyTorch k-means clustering")

path = "/pfs/work7/workspace/scratch/ku4408-VL_ScalableAI/data/cityscapes_300.h5"
dataset = "cityscapes_data"

if rank==0: 
    print("Loading data... {}[{}]".format(path, dataset), end="")
    print("\n")
    
with h5py.File(path, "r") as handle:
    chunk = int(handle[dataset].shape[0]/size)
    if rank==size-1: data = torch.tensor(handle[dataset][rank*chunk:])
    else: data = torch.tensor(handle[dataset][rank*chunk:(rank+1)*chunk])

print("\t[OK]")

# k-means parameters
num_clusters = 8
num_iterations = 20

kmeans = KMeans(n_clusters=num_clusters, max_iter=num_iterations)
if rank==0: 
    print("Start fitting the data...")
    start = time.perf_counter() # Start runtime measurement.
    
kmeans.fit(data)            # Perform actual k-means clustering.

if rank==0: 
    end = time.perf_counter()   # Stop runtime measurement.
    print("DONE.")
    print("Run time:","\t{}s".format(end - start), "s")