mpirun -np 8 vtune -collect hotspots -result-dir vtune_results ./exe_pll

#vtune-gui vtune_results
