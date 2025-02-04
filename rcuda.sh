nvcc -o exe_cuda main_MPI_cuda.cu -I/usr/lib/x86_64-linux-gnu/openmpi/include/ -L/usr/lib/x86_64-linux-gnu/openmpi/lib -Xcompiler -fopenmp -lcusolver -lcublas -llapacke -lblas -lm -lmpi



#mpicc -o exe_cuda main_MPI_cuda.cu -lm -llapacke -lblas -lcusolver -lcublas -fopenmp
#mpirun -np 5 ./exe_pll 



#mpicc -o exe main_MPI.c -lm -L/usr/lib/x86_64-linux-gnu -lscalapack-openmpi -lblacs-openmpi -llapack -lblas -fopenmp
#./exe #>&out&
