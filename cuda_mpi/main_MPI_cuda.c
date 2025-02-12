#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<complex.h>

#include<omp.h>
#include<mpi.h>

#include "cuda_functions.cu"
#include<cuComplex.h>

#include<cblas.h>
#include<lapacke.h>

#define PI 3.1415926535897932

// Function to calculate potential
double potential(double x){
  return 0.5*x*x;  // Harmonic oscillator
}

// Allocate a dynamic 2D complex array
double complex **calloc_2d_array_complex(int rows, int cols) {
    double complex **array = (double complex**)calloc(rows, sizeof(double complex*));
    
    if (array == NULL) {
        printf("Memory allocation failed for rows!\n");
        exit(1);
    }

    for (int i = 0; i < rows; i++) {
        array[i] = (double complex*)calloc(cols, sizeof(double complex));
        if (array[i] == NULL) {
            printf("Memory allocation failed for row %d!\n", i);
            exit(1);
        }
    }

    return array;
}


cuDoubleComplex **allocateMatrix(int n) {
    cuDoubleComplex **matrix = (cuDoubleComplex **)malloc(n * sizeof(cuDoubleComplex *));
    for (int i = 0; i < n; i++) {
        matrix[i] = (cuDoubleComplex *)calloc(n, sizeof(cuDoubleComplex));
    }
    return matrix;
}


// Free a dynamic 2D complex array
void free_2d_array_complex(double complex** array, int rows) {
    for (int i = 0; i < rows; i++) {
        free(array[i]);
    }
    free(array);
}

void copy_matrix_complex_to_complex(int ncsf, double complex **source, double complex **dest) {
    for (int i = 0; i < ncsf; i++) {
        for (int j = 0; j < ncsf; j++) {
            dest[i][j] = source[i][j];
        }
    }
}

void copy_matrix_real_to_complex(int ncsf, double **source, double complex **dest) {
    for (int i = 0; i < ncsf; i++) {
        for (int j = 0; j < ncsf; j++) {
            __real__ dest[i][j] = source[i][j];
            __imag__ dest[i][j] = 0.0;
        }
    }
}


int main(){
  int ntim, ncstep, istate, flqchnl;
  int ncsf, noptc;
  double xmin, xmax, epsln, omga;
  double tau, totime;

  ncsf = 2000; flqchnl = 3; xmin = 0 ; xmax = 10.0;
  epsln = 0.5; omga = 0.5;
  noptc = 5; istate = 1;                             // keep istate as 1, 0 giving wrong result
  
  tau = 2.0*PI/omga;
  totime = noptc*tau;
  ntim = (int)totime;

  MPI_Init(NULL, NULL);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double startTime, endTime, serialTime, loopTime;
  double st, et, tot;

  double mass = 10000.0; 
  
  double dtim = totime/ntim;
  
  double dx = (xmax-xmin)/(ncsf-1);
	   
  int tchnl = 2*flqchnl+1;
  
  int nfloq = ncsf*tchnl;
  
  int inzr = flqchnl*ncsf;

  int npar = tchnl/size;
  int nres = tchnl%size;

  double complex alpha = 1.0 + 0.0*I;
  double complex beta = 0.0 + 0.0*I;

  double complex **exhm = calloc_2d_array_complex(ncsf, ncsf);

  double complex* lexhm = (double complex*)calloc(ncsf*ncsf, sizeof(double complex));

  double complex* cofh = (double complex*)calloc(ncsf, sizeof(double complex));
  double complex* cofc = (double complex*)calloc(ncsf, sizeof(double complex));
  
  // --------------- cuda version ---------------
  cuDoubleComplex *h11, *h22, *h33;
  h11 = (cuDoubleComplex *)calloc(ncsf*ncsf, sizeof(cuDoubleComplex));
  h22 = (cuDoubleComplex *)calloc(ncsf, sizeof(cuDoubleComplex));
  h33 = (cuDoubleComplex *)calloc(ncsf, sizeof(cuDoubleComplex));
  // ---------------------------------------------

  if (rank == 0){
        
        if (size == tchnl){
              printf("Process exiting: size == tchnl (%d == %d)\n", size, tchnl);
              fflush(stdout); // Ensure message is printed before exit
              MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); // Ensures all processes exit
        };

        double finalstarttime = omp_get_wtime();
        

        printf("ncsf = %d \n", ncsf);
        printf("dx = %lf \n", dx);
        printf("length = %lf \n", (xmax-xmin));
        printf("dt = %lf \n", dtim);
        printf("totime = %lf \n", totime);
        printf("tau = %lf \n", tau);
        printf("omega = %lf \n", omga);
        printf("epsln = %lf \n", epsln);
        printf("tchnl = %d \n", tchnl);
        printf("tau = %lf \n", tau);
        printf("ntim = %d \n", ntim);
        printf("npar = %d \n", npar);        
        printf("nres = %d \n", nres);

        FILE *file;

        double **hmt = (double**)calloc(ncsf,sizeof(double*));
        
        for (int i=0; i<ncsf; i++){
          hmt[i] = (double*)calloc(ncsf,sizeof(double));
        }
  
        // Kinetic energy diagonal part
        
        for(int k=0; k<ncsf; k++){
          hmt[k][k] = (PI*PI)/(6.0*dx*dx);
        }

        // Kinetic energy off-diagonal part
        for(int k=0; k<ncsf; k++){      
          for(int j=0; j<k; j++){	    
            hmt[k][j] = (pow(-1.0,(k-j)))/((k-j)*(k-j)*dx*dx);
            hmt[j][k] = hmt[k][j];
          }
        }

        for(int i=0; i<ncsf; i++){
          for(int j=0; j<ncsf; j++){
	            hmt[i][j] = hmt[i][j]/10000.0;
          }	  
        }
                  
        double *zdipole, *cpm;  
        zdipole = (double *)calloc(ncsf,sizeof(double));
        cpm = (double *)calloc(ncsf,sizeof(double));
      
        for(int i=0; i<ncsf; i++){
          double x = xmin + i*dx;
          zdipole[i] = x;
           
          hmt[i][i] = hmt[i][i] + potential(x);
      
          if (abs(x) > (xmax-30.0)){
            cpm[i] = (abs(x)-(xmax-30.0))*(abs(x)-(xmax-30.0));
          }
        }
        
        double *h; 
        h = (double *)calloc(ncsf*ncsf,sizeof(double));
        
        for(int i=0; i<ncsf; i++){
          for(int j=0; j<ncsf; j++){
          	h[i*ncsf+j] = hmt[i][j];
          }	  
        }
        
        double *eig = (double*)calloc(ncsf,sizeof(double));  
        int info;
        
        double *hmtt = (double*)calloc(ncsf*ncsf,sizeof(double));
        
        startTime = omp_get_wtime();
        // diagonalize matrix using CuSolver
        cuda_diagonalize_matrix(h, ncsf, hmtt, eig);
        endTime = omp_get_wtime();
        
        serialTime = endTime - startTime;
        printf("Time taken in cuSolver diagonalisation = %lf sec\n", serialTime); 
        
        for(int i=0; i<ncsf; i++){
          for(int j=0; j<ncsf; j++){
            hmt[i][j] = hmtt[i+j*ncsf];
          }
        }
        
      
        double complex *cofi = (double complex*)calloc(nfloq, sizeof(double complex));
      
        // skip inzr rows and insert values upto ncsf rows
        for (int i=0; i<ncsf; i++){
          cofi[inzr+i] = hmt[i][istate]; 
        }  
      
        double complex **cumt1 = calloc_2d_array_complex(ncsf, ncsf);
        double complex **cumt2 = calloc_2d_array_complex(ncsf, ncsf);
      
        double complex com1 = -I*dtim;  
        
        for(int i=0; i<ncsf; i++){
          for(int j=0; j<ncsf; j++){
            cumt1[i][j] = cexp(com1*eig[i])*hmt[j][i];
          }  
        }
       
        free(eig);
      
        copy_matrix_real_to_complex(ncsf, hmt, cumt2); 
        
        cuDoubleComplex *h1, *h2, *h3;
        h1 = (cuDoubleComplex *)calloc(ncsf*ncsf,sizeof(cuDoubleComplex));
        h2 = (cuDoubleComplex *)calloc(ncsf*ncsf,sizeof(cuDoubleComplex));
        h3 = (cuDoubleComplex *)calloc(ncsf*ncsf,sizeof(cuDoubleComplex));

        for(int i=0; i<ncsf; i++){
          for(int j=0; j<ncsf; j++){
           	h1[i+j*ncsf] = make_cuDoubleComplex(creal(cumt1[i][j]), cimag(cumt1[i][j]));
            h2[i+j*ncsf] = make_cuDoubleComplex(creal(cumt2[i][j]), cimag(cumt2[i][j]));
          }	  
        }
          
        // lda, ldb, ldc are num_rows_A, num_rows_B and num_rows_C respectively
        startTime = omp_get_wtime(); 

        cuda_complexMatrixMultiply(h2, h1, h3, ncsf);
        endTime = omp_get_wtime(); 
      
        serialTime = endTime - startTime;
        printf("Time taken in cuBlas matrix multiplication 1 = %lf sec\n", serialTime); 
      
        for(int i=0; i<ncsf; i++){
          for(int j=0; j<ncsf; j++){
            exhm[i][j] = CMPLX(cuCreal(h3[i+j*ncsf]), cuCimag(h3[i+j*ncsf]));
          }
        }

        for(int i=1; i<size; i++){
          MPI_Send(h3, ncsf*ncsf, MPI_DOUBLE_COMPLEX, i, 1, MPI_COMM_WORLD);
        } 
      
     
        free_2d_array_complex(cumt1, ncsf); free_2d_array_complex(cumt2, ncsf);
      
      
        for (int i = 0; i < ncsf; i++) {
              free(hmt[i]);
        }
        free(hmt);
      
        //-------------------------------------------------------------------------
        cumt1 = calloc_2d_array_complex(tchnl, tchnl);
        int findx;
        
        for(int i = 0; i<tchnl; i++){
          for(int j = 0; j<tchnl; j++){
            findx = -(tchnl + 1)/2 + (i+1);
            com1 = -I*omga*0.5*dtim*findx;
            cumt1[i][j] = sqrt(2.0/(tchnl+1.0))*sin((i+1)*(j+1)*PI/(tchnl+1.0))*cexp(com1);
          }
        }
        
        // ================== Time Propagation starts =======================
        double complex* coff = (double complex*)calloc(nfloq, sizeof(double complex));
      
        com1 = -0.5*I*dtim;
        double time, epls, eff;
        int n1;
        double complex csum, dipole;
      
      
        FILE *timefile = fopen("./result/finalresult_parallel.txt","w"); 
      
      
        // ------------- time propagation loop starts ---------------- //
      
        
        for(int itim = 0; itim<ntim; itim++){
            startTime = omp_get_wtime(); 
      
            time = itim*dtim;
            epls = epsln*sin((PI*time)/totime)*sin((PI*time)/totime);
                
            for (int i=0; i<tchnl; i++){
              findx = -(tchnl + 2)/2 + i;
              for (int j=0; j<ncsf; j++){
                  n1 = i*ncsf+j;
                  coff[n1] = cofi[inzr+j]*cexp(findx*I*omga*time);
              }
            }
      
        // --------------------------------------------- 
      
            n1 = 0; eff = 0.0;
            for (int i = 0; i < nfloq; i++) cofi[i] = CMPLX(0.0,0.0);
      
            for (int i=0; i<tchnl; i++){
              eff = epls*cos((i+1)*PI/(tchnl+1)); 
              for (int j=0; j<ncsf; j++){
                  csum = CMPLX(0.0,0.0);  
                  for (int k=0; k<tchnl; k++){
                      csum = csum + cumt1[k][i]*coff[ncsf*k+j];
                  }
                  cofi[n1] = csum*cexp(com1*eff*zdipole[j]);
                  n1++;
              }
            }
            


        // ----------- Start of exhm parallel implementation ------------
  
            for (int i = 0; i < nfloq; i++) coff[i] = CMPLX(0.0,0.0);

            int k=0;
            for(int ipar=0; ipar<npar; ipar++){
              k = ipar*size*ncsf;
              
              for(int i = 0; i < ncsf; i++) cofh[i] = CMPLX(0.0,0.0);

              for(int i=0; i<ncsf; i++){
                cofh[i] = cofi[k+i];
              } 

              for(int i = 0; i < ncsf; i++) cofc[i] = CMPLX(0.0,0.0);
              
              // ------------ cuda version ------------------------------
              for(int j=0; j<ncsf; j++){
                 h22[j] = make_cuDoubleComplex(creal(cofh[j]), cimag(cofh[j])); 
              }
              
              for(int i=0; i<ncsf; i++){
                 for(int j=0; j<ncsf; j++){
                	     h11[j*ncsf+i] = make_cuDoubleComplex(creal(exhm[i][j]), cimag(exhm[i][j]));
                 }	  
              }
              
              for (int i = 0; i < ncsf; i++) h33[i] = make_cuDoubleComplex(0.0,0.0);

              cuda_complexMatrixMultiply_col(h11, h22, h33, ncsf);

              for(int i=0; i<ncsf; i++){
                  cofc[i] = CMPLX(cuCreal(h33[i]), cuCimag(h33[i]));
              }
              
              // ---------------------------------------------------

              for(int i=0; i<ncsf; i++){
                coff[k+i] = cofc[i];
              }

             // ------------ send to other processes --------------
              for(int i=1; i<size; i++){
                n1 = k + i*ncsf;
                for(int j = 0; j < ncsf; j++) cofh[j] = CMPLX(0.0,0.0);
                
                for(int j=0; j<ncsf; j++){
                  cofh[j] = cofi[n1+j];
                }  

                
                MPI_Send(cofh, ncsf, MPI_DOUBLE_COMPLEX, i, 1, MPI_COMM_WORLD);
              } 

              
              for(int i=1; i<size; i++){
                n1 = k + i*ncsf;

              
                MPI_Recv(cofc, ncsf, MPI_DOUBLE_COMPLEX, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                
                for(int j=1; j<ncsf; j++){
                  coff[n1 + j] = cofc[j];
                }            
              } 
            }



            for (int i = 0; i < ncsf; i++) cofh[i] = CMPLX(0.0,0.0);
            n1 = ncsf*npar*size;

            for(int i=0; i<ncsf; i++){
              cofh[i] = cofi[n1 + i];
            }

            for (int i = 0; i < ncsf; i++) cofc[i] = CMPLX(0.0,0.0);

            // -----------cuda version----------------------------
               for(int j=0; j<ncsf; j++){
                 h22[j] = make_cuDoubleComplex(creal(cofh[j]), cimag(cofh[j])); 
              }
              
              for(int i=0; i<ncsf; i++){
                 for(int j=0; j<ncsf; j++){
                	     h11[j*ncsf+i] = make_cuDoubleComplex(creal(exhm[i][j]), cimag(exhm[i][j]));
                 }	  
              }
              
              for (int i = 0; i < ncsf; i++) h33[i] = make_cuDoubleComplex(0.0,0.0);

              cuda_complexMatrixMultiply_col(h11, h22, h33, ncsf);

              for(int i=0; i<ncsf; i++){
                  cofc[i] = CMPLX(cuCreal(h33[i]), cuCimag(h33[i]));
              }
              // -----------------------------------------------------


            for(int i=0; i<ncsf; i++){
               coff[n1 + i] = cofc[i];
            } 

            for(int i=1; i<nres; i++){
                for (int i = 0; i < ncsf; i++) cofh[i] = CMPLX(0.0,0.0);
                
                for(int j=0; j<ncsf; j++){
                   cofh[j] = cofi[n1 + i*ncsf + j];
                }
                //correct
                MPI_Send(cofh, ncsf, MPI_DOUBLE_COMPLEX, i, 1, MPI_COMM_WORLD); 
            } 
            
            for(int i=1; i<nres; i++){
                // correct
                MPI_Recv(cofc, ncsf, MPI_DOUBLE_COMPLEX, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for(int j=0; j<ncsf; j++){
                  coff[n1 + i*ncsf + j] = cofc[j];
                }        
            }
     
            // ----------------End of exhm---------------------
            
            n1 = 0; eff = 0;
            for (int i = 0; i < nfloq; i++) cofi[i] = CMPLX(0.0,0.0);
      
            for (int i=0; i<tchnl; i++){
                eff = epls*cos((i+1)*PI/(tchnl+1));
                for (int j=0; j<ncsf; j++){
                    cofi[n1] = cexp(com1*zdipole[j]*eff)*coff[n1];
                    n1++;
                }
            }
             
            n1 = 0;
            for (int i = 0; i < nfloq; i++) coff[i] = CMPLX(0.0,0.0);
      
            for (int j=0; j<tchnl; j++){
                for (int i=0; i<ncsf; i++){
                    csum=CMPLX(0.0,0.0);  
                    for (int k=0; k<tchnl; k++){
                       csum = csum + cumt1[j][k]*cofi[ncsf*k+i];
                    }
                    coff[n1] = csum;
                    n1++;
                }
            }
      
            for (int i = 0; i < nfloq; i++) cofi[i] = CMPLX(0.0,0.0);
      
            for (int i=0; i<nfloq; i++){
              cofi[i] = coff[i];
            }
            
            for (int i = 0; i < ncsf; i++) cofh[i] = CMPLX(0.0,0.0);
      
            for (int i=0; i<ncsf; i++){
              cofh[i] = coff[inzr+i];
            }
      
            dipole = CMPLX(0.0,0.0);
            for (int i=0; i<ncsf; i++){
              dipole += conj(cofh[i])*epls*zdipole[i]*cos(omga*time)*cofh[i];
            }
      
            fprintf(timefile,"%e %e %e %e \n",time/tau, epls*cos(omga*time), creal(dipole), cimag(dipole)); 
        
            endTime = omp_get_wtime();
            loopTime = endTime - startTime;
            printf("Time taken in %d loop = %lf sec\n", itim, loopTime); 

        
        }   // time loop ends
        fclose(timefile);  
         
        double finalendtime = omp_get_wtime();
        double finaltime = finalendtime - finalstarttime;
        printf("Final time-taken by CUDA = %lf sec\n", finaltime); 
        

        free(coff); free(cofi); 
        free(zdipole);
        free_2d_array_complex(cumt1, tchnl); 
        
  } // mpi rank=0 ends
  else{
        
        MPI_Recv(lexhm, ncsf*ncsf, MPI_DOUBLE_COMPLEX, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int i=0; i<ncsf; i++){
            for(int j=0; j<ncsf; j++){
                exhm[i][j] = lexhm[j*ncsf+i];
            }
        }

        for(int itim = 0; itim<ntim; itim++){
            for(int ipar = 0; ipar<npar; ipar++){ 

                //correct 
                MPI_Recv(cofh, ncsf, MPI_DOUBLE_COMPLEX, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // --------- cuda version ------------------------------

                for(int j=0; j<ncsf; j++){
                    h22[j] = make_cuDoubleComplex(creal(cofh[j]), cimag(cofh[j])); 
                }
              
                for(int i=0; i<ncsf; i++){
                  for(int j=0; j<ncsf; j++){
                	     h11[j*ncsf+i] = make_cuDoubleComplex(creal(exhm[i][j]), cimag(exhm[i][j]));
                  }	  
                }
              
                for (int i = 0; i < ncsf; i++) h33[i] = make_cuDoubleComplex(0.0,0.0);
       

                cuda_complexMatrixMultiply_col(h11, h22, h33, ncsf);

                for(int i=0; i<ncsf; i++){
                    cofc[i] = CMPLX(cuCreal(h33[i]), cuCimag(h33[i]));
                }
                // -----------------------------------------

                MPI_Send(cofc, ncsf, MPI_DOUBLE_COMPLEX, 0, 2, MPI_COMM_WORLD);
                
            }      
           
            if (rank < nres){ 
                MPI_Recv(cofh, ncsf, MPI_DOUBLE_COMPLEX, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);                
 
                for (int i = 0; i < ncsf; i++) cofc[i] = CMPLX(0.0,0.0);
              
              // ------------cuda version ------------------------------ 
                for(int j=0; j<ncsf; j++){
                   h22[j] = make_cuDoubleComplex(creal(cofh[j]), cimag(cofh[j])); 
                }
                
                for(int i=0; i<ncsf; i++){
                   for(int j=0; j<ncsf; j++){
                  	     h11[j*ncsf+i] = make_cuDoubleComplex(creal(exhm[i][j]), cimag(exhm[i][j]));
                   }	  
                }
                
                for (int i = 0; i < ncsf; i++) h33[i] = make_cuDoubleComplex(0.0,0.0);
  
                cuda_complexMatrixMultiply_col(h11, h22, h33, ncsf);
  
                for(int i=0; i<ncsf; i++){
                    cofc[i] = CMPLX(cuCreal(h33[i]), cuCimag(h33[i]));
                }
                
                // ------------------------------------------
                
                MPI_Send(cofc, ncsf, MPI_DOUBLE_COMPLEX, 0, 2, MPI_COMM_WORLD);
            }
            
        } // time loop for slaves ends

  }// MPI rank != 0 ends


  // free memory
  free_2d_array_complex(exhm, ncsf); 
  free(cofh); free(cofc);
  free(h11); free(h22); free(h33);


  MPI_Finalize();

  return 0;
}
