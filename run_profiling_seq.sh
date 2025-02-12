vtune -collect hotspots -result-dir vtune_results ./exe_seq

vtune-gui vtune_results
