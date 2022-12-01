#!/bin/bash

#SBATCH --job-name=U2_A2(No duplicates)    # job name
#SBATCH --partition=multiple               # queue for the resource allocation.
#SBATCH --time=5:00                        # wall-clock time limit  
#SBATCH --mem=90000                        # memory per node
#SBATCH --nodes=4                          # number of nodes to be used
#SBATCH --cpus-per-task=40                 # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --mail-user=uuqkm@student.kit.edu  # notification email address

export IBV_FORK_SAFE=1
module purge                                     # Unload all currently loaded modules.
module load compiler/gnu/11.2                    # Load required modules.  
module load mpi/openmpi/4.1
module load lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1
source A2/bin/activate   # Activate your virtual environment.

mpirun --mca mpi_warn_on_fork 0 python A2.py no_duplicates # Adjust path to your psrs python script.
                                                                 # Specify dataset via command-line argument.  