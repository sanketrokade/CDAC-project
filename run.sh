gcc -o exe_seq main_timing.c -lm -llapacke -llapack -lblas -fopenmp
./exe_seq
#./exe_seq >&seq_out&
