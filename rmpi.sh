mpicc -o exe_pll main_MPI.c -lm -llapacke -lblas -fopenmp
mpirun -np 5 ./exe_pll 



#mpicc -o exe main_MPI.c -lm -L/usr/lib/x86_64-linux-gnu -lscalapack-openmpi -lblacs-openmpi -llapack -lblas -fopenmp
#./exe #>&out&






#nvcc -c kernel_code.cu -o kernel_code.o -lcusolver -lcublas -I/usr/local/cuda-12.8/include/
#mpicc -c main_MPI_cuda.c -o exe.o -fopenmp -lm -lblas -llapacke -I/usr/local/cuda-12.8/include/

#mpicc -o exe_cuda exe.o kernel_code.o -lcusolver -lcublas -lcudart -llapacke -lblas -lm -lmpi -fopenmp -L/usr/local/cuda/lib64
