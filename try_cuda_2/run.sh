nvcc -c cuda_functions.cu -o cuda_functions.o -lcusolver -lcublas -I/usr/local/cuda-12.8/include/

mpicc -c main_MPI_cuda.c -o exe.o -fopenmp -lm -lblas -llapacke -I/usr/local/cuda-12.8/include/

mpicc -o exe_cuda exe.o cuda_functions.o -lcusolver -lcublas -lcudart -llapacke -lblas -lm -lmpi -fopenmp -L/usr/local/cuda-12.8/lib64

mpirun -np 5 ./exe_cuda
