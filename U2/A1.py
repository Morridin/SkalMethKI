import torch
import numpy as np
import time
from mpi4py import MPI
import h5py 
torch.set_default_tensor_type(torch.FloatTensor)

def dist(X, Y, comm=MPI.COMM_WORLD):
    """
    Pairwise distance calculation between all elements along axis 0 of X and Y. 
    Returns 2D torch.tensor of size m x n.
    Distance matrix is calculated tile-wise with ring communication between processes
    holding each a piece of X and/or Y.

    Parameters
    ----------
    X : torch.Tensor
        2D Array of size m/p x f.
    Y : torch.Tensor
        2D array of size n/p x f.
    comm : MPI communicator
           communicator to use
    """
    size = comm.size
    rank = comm.rank

    if X.shape[1] != Y.shape[1]: raise ValueError("Inputs must have same # of features.")
    
    if size == 1:
        return torch.cdist(X, Y)

    else:
        mp, f = X.shape
        np = Y.shape[0]
        
        # Each rank has distance matrix chunk of size mp x n, i.e., rank 0 has distances 
        # from its own local X to all other Y's.
        n = comm.allreduce(np, op = MPI.SUM)
        print("Overall number of samples is", n)
        ycounts = torch.tensor(comm.allgather(torch.numel(Y)//f), dtype=torch.int)
        ydispl = (0,) + tuple(torch.cumsum(ycounts, dim=0,dtype=torch.int)[:-1])
        dl = torch.zeros((mp, n))   # Initialize rank-local chunk of distance matrix with zeros.
        cols = (ydispl[rank], ydispl[rank + 1] if (rank + 1) != size else n)
        
        x_ = X
        stationary = Y
        
        # 0th iteration: Calculate diagonal.
        print(f"Rank [{rank}/{size}]: Start calculating diagonals...")
        d_ij = torch.cdist(x_, stationary)
    
        dl[:, cols[0] : cols[1]] = d_ij
        
        print(f"Rank [{rank}/{size}]: Start tile-wise ring communication...")
        for iter in range(1, size):
            # Send rank's part of matrix to next process in circular fashion.
            receiver = (rank + iter) % size
            sender = (rank - iter) % size
            col1 = ydispl[sender]
            col2 = ydispl[sender + 1] if sender != size - 1 else n
            columns = (col1, col2)
            # All but first iter processes are receiving, then sending.
            if (rank // iter) != 0:
                stat = MPI.Status()
                comm.Probe(source=sender, tag=iter, status=stat)
                count = int(stat.Get_count(MPI.FLOAT) / f)
                moving = torch.zeros((count, f))
                comm.Recv(moving, source=sender, tag=iter)
            # Sending to next process.
            comm.Send(stationary, dest=receiver, tag=iter)
            # First iter processes can now receive after sending.
            if (rank // iter) == 0:
                stat = MPI.Status()
                comm.Probe(source=sender, tag=iter, status=stat)
                count = int(stat.Get_count(MPI.FLOAT) / f)
                moving = torch.zeros((count, f))
                comm.Recv(moving, source=sender, tag=iter)
            d_ij = torch.cdist(x_, moving)
            dl[:, columns[0] : columns[1]] = d_ij
        print(f"Rank [{rank}/{size}]: [DONE]")
        return dl

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

path = "/pfs/work7/workspace/scratch/ku4408-VL_ScalableAI/data/SUSY_50k.h5"
dataset = "data"

if rank == 0:
    print("######################")
    print("# Pairwise distances #")
    print("######################")
    print(f"COMM_WORLD size is {size}.")
    print(f"Loading data... {path}[{dataset}]")

# Parallel data loader for SUSY data.
with h5py.File(path, "r") as handle:
    chunk = int(handle[dataset].shape[0]/size)
    if rank == size - 1: 
        data = torch.FloatTensor(handle[dataset][rank*chunk:])
    else: 
        data = torch.FloatTensor(handle[dataset][rank*chunk:(rank+1)*chunk])

print("\t[OK]")
print(f"Rank [{rank}/{size}]: Local data chunk has shape {list(data.shape)}...")

if rank == 0: 
    print("Start distance calculations...")
start = time.perf_counter()
d = dist(data, data, comm)
end = time.perf_counter()
run = end - start
Run = comm.allreduce(run, op=MPI.SUM)
Run = Run / size
print(f"Rank [{rank}/{size}]: Local distance matrix has shape {list(d.shape)}.")
if rank == 0: 
    print(f"Run time:\t{Run} s")