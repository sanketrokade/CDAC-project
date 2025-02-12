nvcc -arch=sm_86 -Wno-deprecated-gpu-targets -g -G -c cuda_functions.cu -o cuda_functions.o -lcusolver -lcublas -I/usr/local/cuda-12.8/include/

mpicc -g -O2 -c main_MPI_cuda.c -o exe.o -fopenmp -lm -lblas -llapacke -I/usr/local/cuda-12.8/include/

mpicc -o exe_cuda exe.o cuda_functions.o -lcusolver -lcublas -lcudart -llapacke -lblas -lm -lmpi -fopenmp -L/usr/local/cuda-12.8/lib64

nsys profile --trace=cuda,nvtx,mpi,openmp -o profile_output mpirun -np 8 ./exe_cuda

#nsys profile --stats=true --trace=cuda,nvtx,mpi,openmp -o profile_output mpirun -np 8 ./exe_cuda


#-g -G (for nvcc) → Enables debugging info for CUDA
#-lineinfo (for nvcc) → Shows file and line numbers in profiling
#-g (for mpicc) → Enables debugging info for MPI code
#-O2 → Optimizations enabled (removes -O3 to balance debugging)

