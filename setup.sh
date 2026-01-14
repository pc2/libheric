#!/bin/bash

module reset

# LibERI_ACC
module load fpga
module load intel/oneapi/23.2
module load bittware/520n/20.4.0_hpc
module load intel/oneapi_queue_extensions
module load xilinx/xrt/2.16
module load system/CUDA/12.8.0
module load system/ROCM/6.3.3

# MPI, ScaLAPACK, FFTW, etc. 
module load toolchain/foss/2023b

# Build tools
module load devel/CMake/3.27.6-GCCcore-13.2.0
module load devel/Autoconf/2.71-GCCcore-13.2.0
module load devel/Automake/1.16.5-GCCcore-13.2.0

# Python
module load lang/Python/3.12.3-GCCcore-13.2.0

# Libint2
module load math/Eigen/3.4.0-GCCcore-13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0
module load devel/Boost/1.83.0-GCC-13.2.0

