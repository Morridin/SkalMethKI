import numpy
import time
import torch
import sys
from mpi4py import MPI
import h5py

__mpi_type_mappings = {
        torch.bool: MPI.BOOL,
        torch.uint8: MPI.UNSIGNED_CHAR,
        torch.int8: MPI.SIGNED_CHAR,
        torch.int16: MPI.SHORT,
        torch.int32: MPI.INT,
        torch.int64: MPI.LONG,
        torch.bfloat16: MPI.INT16_T,
        torch.float16: MPI.INT16_T,
        torch.float32: MPI.FLOAT,
        torch.float64: MPI.DOUBLE,
        torch.complex64: MPI.COMPLEX,
        torch.complex128: MPI.DOUBLE_COMPLEX}


def sort(a, comm=MPI.COMM_WORLD):
    """
    Sorts a's elements along given dimension in ascending order by their value.
    The sorting is not stable which means that equal elements in the result may have different ordering than in
    original array.
    Returns tuple (values, indices) with sorted local results and indices of elements in original data.

    Parameters
    ----------
    a : torch tensor
        1D input array to be sorted.
    out : torch tensor, optional
        Location to store results. If provided, it must have broadcastable shape. 
        If not provided or set to `None`, a fresh array is allocated.
    """
    
    size = comm.size
    rank = comm.rank
    
    if size == 1:
        local_sorted, local_indices = torch.sort(a)
        return local_sorted, local_indices
    
    else:
        ###########
        # PHASE 1 #
        ###########
        # Note: p = MPI.COMM_WORLD.size
        #       n = overall # of samples
        # Each rank sorts its local chunk and chooses p regular samples as representatives.
        if rank == 0: 
            print("###########")
            print("# PHASE 1 #")
            print("###########")
        local_sorted, local_indices = torch.sort(a)
        print(f"Rank [{rank}/{size}]: Local sorting done...[OK]")
        
        length = torch.tensor(torch.numel(local_sorted), dtype=torch.int) # Number of elements in local chunk.
        print(f"Rank [{rank}/{size}]: Number of elements in local chunk is {length}.")
        counts = torch.zeros(size, dtype=torch.int)                       # Initialize array for local element numbers.
        comm.Allgather([length, MPI.INT], [counts, MPI.INT])
        displ = (0,) + tuple(torch.cumsum(counts, dim=0)[:-1])
        # Shift indices according to displacements between ranks.
        actual_indices = local_indices + displ[rank]
        
        # Each rank chooses p regular samples.
        # For this, separate sorted tensor into p+1 equal-length partitions.
        # Regular samples have indices 1, w+1, 2w+1,...,(p−1)w+1 
        # where w=n/p^2 (here: 'size' = p, 'length'= overall number of samples/p) 'length' = n/p.
        partitions = [int(x*length / size) for x in range(0, size )]
        
        #for x in range(1, size+1):
        #    print("Partitions: ", partitions)
        reg_samples = local_sorted[partitions]
        if len(partitions) == size: 
            print(f"Rank [{rank}/{size}]: There are {len(partitions)} local regular samples: {reg_samples}")
        
        # Root gathers regular samples.
        num_regs = torch.numel(reg_samples)                  # Get number of local regular samples on each rank.
        regs_dim = int(comm.allreduce(num_regs, op=MPI.SUM)) # Get overall number of regular samples.
        if rank == 0: 
            print(f"Overall number of regular samples is {regs_dim}.")
        regs_global = torch.zeros(regs_dim, dtype=a.dtype)
        comm.Gather(reg_samples, regs_global, root=0)
        if rank == 0: 
            print("On root: Regular samples gathered...[OK]")
        
        ###########
        # PHASE 2 #
        ###########
        # Root sorts gathered regular samples, chooses pivots, and shares them with other processes.
        if rank==0: 
            print("###########")
            print("# PHASE 2 #")
            print("###########")
        global_pivots = torch.zeros((size-1,), dtype=local_sorted.dtype)
        if rank == 0:
            sorted_regs, _ = torch.sort(regs_global)
            print(f"On root: Regular samples are {sorted_regs}.")
            length_regs = sorted_regs.size()[0] # Get overall number of regular samples.
            # Choose p-1 pivot indices (p = MPI size).
            global_partitions = [int(x*length_regs / size) for x in range(1, size)] 
            global_pivots = sorted_regs[global_partitions]
            if len(global_partitions) == size - 1: 
                print(f"On root: There are {len(global_partitions)} global pivots: {global_pivots}")
        # Broadcast copy of pivots to all processes from root.
        comm.Bcast(global_pivots, root=0)
        if rank == 0: 
            print("Pivots broadcasted to all processes...")
        ###########
        # PHASE 3 #
        ###########
        if rank == 0: 
            print("###########")
            print("# PHASE 3 #")
            print("###########")
            print("Each processor forms p disjunct partitions of locally sorted elements using pivots as splits.")
        # Each processor forms p disjunct partitions of locally sorted elements using p-1 pivots as splits.
        lt_partitions = torch.empty((size, local_sorted.shape[0]), dtype=torch.int64)
        last = torch.zeros_like(local_sorted, dtype=torch.int64)
        # Iterate over all pivots and store index of first pivot greater than element's value
        if rank == 0: 
            print("Iterate over pivots to find index of first pivot > element's value.") 

        for idx, p in enumerate(global_pivots):
            # torch.lt(input, other, *, out=None) computes input<other element-wise.
            # Returns boolean tensor that is True where input is less than other and False elsewhere.
            lt = torch.lt(local_sorted, p).int()
            if idx > 0:
                lt_partitions[idx] = lt-last
            else:
                lt_partitions[idx] = lt
            last = lt
        lt_partitions[size-1] = torch.ones_like(local_sorted, dtype=last.dtype)-last

        # lt_partitions contains p elements, first encodes which elements 
        # in local_sorted are smaller than 1st (= smallest) pivot, second encodes which 
        # elements are larger than 1st and smaller than 2nd pivot, ...,
        # last elements encodes which elements are larger than last ( = largest) pivot.
        # Now set up matrix holding info how many values will be sent for each partition.
        # Processor i keeps ith partitions and sends jth partition to processor j.
        local_partitions = torch.sum(lt_partitions, dim=1)    # How many values will be send where (local)?
        print(f"Rank [{rank}/{size}]: Local # elements to be sent to rank [0, 1, 2, 3] (keep own section): {local_partitions}")
        partition_matrix = torch.zeros_like(local_partitions) # How many values will be send where (global)?
        comm.Allreduce(local_partitions, partition_matrix, op=MPI.SUM)
        if rank == 0: 
            print(f"Global # of elements on rank [0, 1, 2, 3] (partition matrix): {partition_matrix}")
        # Matrix holding info which value will be shipped where.
        index_matrix = torch.empty_like(local_sorted, dtype=torch.int64)
        # Loop over lt_partitions (binary encoding of which elements is in which partition formed by pivots)
        for i, x in enumerate(lt_partitions):
            index_matrix[x > 0] = i 
            # Elements in 0th partition (< first pivot) get 0, i.e., will be collected at rank 0,
            # elements in 1st partition (> than first + < than second pivot) get 1, i.e., will
            # be collected at rank 1,...
        print(f"Rank [{rank}/{size}]: Ship element to rank: {index_matrix}")
        scounts = numpy.zeros(size, dtype=int)
        rcounts = numpy.zeros(size, dtype=int)
        el = local_sorted.element_size()
        for s in numpy.arange(size):
            scounts[s] = int((index_matrix == s).sum(dim=0))
        scounts = scounts
        sdispl = numpy.zeros(size, dtype=int)
        sdispl[1:] = numpy.cumsum(scounts, axis=0)[:-1]
        Scounts = numpy.zeros((size, size), dtype=int)
        comm.Allgather([scounts, MPI.INT], [Scounts, MPI.INT])
        Rcounts = numpy.transpose(Scounts)
        rcounts = Rcounts[rank]
        rdispl = numpy.zeros(size, dtype=int)
        rdispl[1:] = numpy.cumsum(rcounts, axis=0)[:-1]
        # Counts + displacements for Alltoallv rank-specific.
        # send_counts on rank i: Integer array where entry j specifies how many values are to be sent to rank j.
        # recv_counts on rank i: Integer array where entry j specifies how many values are to be received from rank j.
        val_buf = torch.zeros((partition_matrix[rank],), dtype = local_sorted.dtype)
        idx_buf = torch.zeros((partition_matrix[rank],), dtype = local_indices.dtype)
        scounts = scounts.tolist()
        rcounts = rcounts.tolist()
        # Element-wise displacements.
        sdispl = sdispl.tolist()
        rdispl = rdispl.tolist()
        
        sbuf =  [MPI.memory.fromaddress(local_sorted.data_ptr(),0), (scounts, sdispl),__mpi_type_mappings[local_sorted.dtype]]
        rbuf = [MPI.memory.fromaddress(val_buf.data_ptr(),0),(rcounts, rdispl),__mpi_type_mappings[val_buf.dtype]]
        comm.Alltoallv(sbuf, rbuf)
        res,_ = torch.sort(val_buf)
        return res

path = "/pfs/work7/workspace/scratch/ku4408-VL_ScalableAI/data/psrs_data.h5"
dataset = str(sys.argv[1]) # 'duplicates_1', 'duplicates_10', 'duplicates_5', 'many_duplicates', 'no_duplicates'

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank

with h5py.File(path, "r") as f:
    chunk = int(f[dataset].shape[0]/size)
    if rank == size - 1: 
        data = torch.tensor(f[dataset][rank*chunk:])
    else: 
        data = torch.tensor(f[dataset][rank*chunk:(rank+1)*chunk])

if rank == 0:
    print("########")
    print("# PSRS #")
    print("########")

print(f"Local data on rank {rank} = {data}")

if rank == 0: 
    print("Start sorting...")
start = time.perf_counter()
res = sort(data)
end = time.perf_counter()
run = end - start
Run = comm.allreduce(run, op=MPI.SUM)
Run = Run / comm.size
if rank == 0: 
    print("Sorting done...")
    print(f"Rank-averaged run time: {Run} s")
print(res)