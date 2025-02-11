## Optimization of Scientific Numerical Code using Parallel Programming

- This project aims to optimize the performance of a sequential program by utilizing parallel programming techniques such as MPI, openMP and GPU programming using CUDA. The performance of each parallel programming technique will be evaluated and compared.

### The sequential code
- The sequential C code implements the time-propagation of a quantum mechanical wavefunction in a harmonic potential in the presence of a laser using numerical linear algebra methods. It involves matrix operations, eigenvalue solving, and time evolution algorithm commonly used in quantum mechanics simulations. The key features of the code include constructing a Hamiltonian matrix, diagonalizing it, and propagating a wavefunction as initial state using matrix exponentials in the context of the Floquet formalism.

- The key components in the code are:
  - Potential Function(`potential`):
    - Defines a harmonic oscillator potential for a quantum particle $V(x) = \frac{1}{2}x^{2}$ (i.e., a quadratic potential) for position x.

    - Main computational Function: `timeprop`
      - Inputs for the program:
        - `ncsf`: Number of spatial grid points.
        - `xmin, xmax`: Boundaries of the 1D spatial domain.
        - `lmda, flqchnl, noptc, istate, totime, ntim, omga, epsln`: Parameters for time-propagation, Floquet channels, and external driving.
          - `lmda`: Magnitude of the Complex Absorbing Potential
          - `flqchnl`: Number of floquet channels
          - `noptc`: Number of optical cycles
          - `istate`: State index of the eigenfunction to be used as initial wavefunction
          - `totime`: Total time of the time-propagation
          - `ntim`: Number of time propagation loops
          - `omga`: Frequency of laser
          - `epsln`: Electric field strength of the laser 

      - Key Steps in `timeprop`:
        - Discretization of Space and time: 
          - The space is discretized into `ncsf` points between `xmin` and `xmax` with spacing `dx`.
          - Time is divided into `ntim` steps of size `dtim`.
        - Kinetic Energy Matrix (hmt):
          - Constructed using the second derivative approximation in finite difference:
            $$T_{i,j} = \frac{\pi^2}{6 dx^{2}}, \text{for } i = j$$<br> 
            $$T_{i,j} = \frac{(-1)^{i-j}}{(i-j)^2 dx^{2}}, \text{for } i \ne j$$ 
        - Potential Energy: 
          - The diagonal of `hmt` is modified to include the harmonic potential:
            $$H_{ii} = T_{ii} + V{x_i}$$
        
        - Diagonalization of Hamiltonian:
          - LAPACK's LAPACKE_dsyev computes eigenvalues and eigenvectors of hmt.
          - Eigenvalues represent energy levels, and eigenvectors represent stationary states.

        - Wavefunction Initialization:
          - The wavefunction is initialized using the eigenvector corresponding to the specified initial state (`istate`).

        - Floquet Matrix Setup:
          - The code creates Floquet channel matrices (`cumt1`, `cumt2`) for time propagation.

          - These matrices are used to handle periodic driving fields.

        - Matrix Exponentials:
          - The core of time propagation involves matrix exponentials:
            - Constructed from eigenvalues for time evolution.
            - Includes external field effects via Floquet formalism.
        
        - BLAS Operations:
          - BLAS routines (`cblas_zgemm`) perform matrix multiplications for efficient computation.

        - Time Evolution:
          - The wavefunction evolves in time using:
            - The Floquet matrices.
            - The complex Hamiltonian propagator (`exhm`).
          - This evolution includes the effects of an oscillating external field.


* Profiling using Nsight Systems
  * `nsys profile --stats=true --trace=cuda,nvtx,mpi,openmp -o profile_output mpirun -np 8 ./exe_cuda`
    * nsys profile → Runs Nsight Systems profiler.
    * --stats=true → Displays a summary of profiling statistics in the terminal after execution.
    * --trace=cuda,nvtx,mpi,openmp → Captures profiling data for:
      * cuda → CUDA kernel launches, memory transfers, etc.
      * nvtx → NVIDIA Tools Extension (for custom profiling markers).
      * mpi → MPI communication profiling.
      * openmp → OpenMP thread execution profiling.

    * -o profile_output → Saves profiling data to a file named profile_output.qdrep.
    * mpirun -np 8 ./exe_cuda → Runs exe_cuda with 8 MPI processes.
  * Analyse in GUI
    * `nsys-ui profile_output.nsys-rep`
  
