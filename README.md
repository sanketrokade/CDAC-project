## Optimization of Scientific Numerical Code using Parallel Programming

* This project aims to optimize the performance of a sequential program by utilizing parallel programming techniques such as MPI, openMP and GPU programming using CUDA. The performance of each parallel programming technique will be evaluated and compared.

### The sequential code
* The sequential C code implements the time-propagation of a quantum mechanical wavefunction in a harmonic potential in the presence of a laser using numerical linear algebra methods. It involves matrix operations, eigenvalue solving, and time evolution algorithm commonly used in quantum mechanics simulations.

* The code simulates the time evolution of a quantum state in a harmonic potential, based on a given initial state and a specified time-dependent interaction. The primary goal is to compute the dipole moment of the system over time.

* The key components in the code are:
  * Potential Function(`potential`):
    * Defines a harmonic oscillator potential for a quantum particle $V(x) = \frac{1}{2}x^{2}$ (i.e., a quadratic potential) for position x.

    * Main computational Function: `timeprop`
      * Inputs for the program:
        * ncsf: Number of spatial grid points.
        * xmin, xmax: Boundaries of the 1D spatial domain.
        * lmda, flqchnl, noptc, istate, totime, ntim, omga, epsln: Parameters for time-propagation, Floquet channels, and external driving.
          * lmda: Magnitude of the Complex Absorbing Potential
          * flqchnl: Number of floquet channels
          * noptc: Number of optical cycles
          * istate: State index of the eigenfunction to be used as initial wavefunction
          * totime: Total time of the time-propagation
          * ntim: Number of time propagation loops
          * omga: Frequency of laser
          * epsln: Electric field strength of the laser 