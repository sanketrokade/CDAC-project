#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<complex.h>
#include<omp.h>


#include<lapacke.h>
#include<cblas.h>

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

// Time propagation function
int timeprop(int ncsf, double xmin, double xmax, int flqchnl, int noptc, 
             int istate, double totime, int ntim, double omga, double epsln){

  double mass = 10000.0; 
  
  double dtim = totime/ntim;

  double dx = (xmax-xmin)/(ncsf-1);
	   
  int tchnl = 2*flqchnl+1;

  int nfloq = ncsf*tchnl;

  int inzr = flqchnl*ncsf;

  double tau = 2.0*PI/omga;

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

  double startTime, endTime, serialTime;

  FILE *file;

  double **hmt = (double**)calloc(ncsf,sizeof(double*));
  
  for (int i=0; i<ncsf; i++){
    hmt[i] = (double*)calloc(ncsf,sizeof(double));
  }


  // Kinetic energy 
  for(int k=0; k<ncsf; k++){
    hmt[k][k] = (PI*PI)/(6.0*dx*dx);

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



  double *zdipole;  
  zdipole = (double *)calloc(ncsf,sizeof(double));

  for(int i=0; i<ncsf; i++){
    double x = xmin + i*dx;
    zdipole[i] = x;
     
    hmt[i][i] = hmt[i][i] + potential(x);
  }


  double *h; 
  h = (double *)calloc(ncsf*ncsf,sizeof(double));
  
  for(int i=0; i<ncsf; i++){
    for(int j=0; j<ncsf; j++){
    	h[i*ncsf+j] = hmt[i][j];
    }	  
  }



  double *eig = (double *)calloc(ncsf,sizeof(double));  
  int info;

  startTime = omp_get_wtime();

  info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', ncsf, h, ncsf, eig);

  endTime = omp_get_wtime();
  
  serialTime = endTime - startTime;
  printf("Time taken in lapack diagonalisation = %lf sec\n", serialTime); 

  if (info != 0) {
      printf("Error in LAPACKE_dsyev: %d\n",info);
//      free(h);
//      free(W);
      return -1;
  }
  
  

  for(int i=0; i<ncsf; i++){
    for(int j=0; j<ncsf; j++){
      hmt[i][j] = h[i*ncsf+j];
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
    double complex exevl = cexp(com1*eig[i]);
    for(int j=0; j<ncsf; j++){
      cumt1[i][j] = exevl*hmt[j][i];
    }  
  }
 
  free(eig);

  copy_matrix_real_to_complex(ncsf, hmt, cumt2); 
  
  double complex **exhm = calloc_2d_array_complex(ncsf, ncsf);

  double complex *h1, *h2, *h3;
  h1 = (double complex *)calloc(ncsf*ncsf,sizeof(double complex));
  h2 = (double complex *)calloc(ncsf*ncsf,sizeof(double complex));
  h3 = (double complex *)calloc(ncsf*ncsf,sizeof(double complex));

  for(int i=0; i<ncsf; i++){
    for(int j=0; j<ncsf; j++){
    	h1[j*ncsf+i] = cumt1[i][j];        // transposing and storing in h1 and h2
      h2[j*ncsf+i] = cumt2[i][j];        // transpose being done to use cblas_zgemm 
    }	  
  }

  double complex alpha = 1.0 + 0.0*I;
  double complex beta = 0.0 + 0.0*I;


// lda, ldb, ldc are num_rows_A, num_rows_B and num_rows_C respectively
  startTime = omp_get_wtime();
  cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              ncsf, ncsf, ncsf, &alpha, h2, ncsf, 
              h1, ncsf, &beta, h3, ncsf);
  endTime = omp_get_wtime();

  serialTime = endTime - startTime;
  printf("Time taken in blas matrix multiplication 1 = %lf sec\n", serialTime); 


// ------ This exhm will be used ahead ---------
  for(int i=0; i<ncsf; i++){
    for(int j=0; j<ncsf; j++){
      exhm[i][j] = h3[j*ncsf+i];
    }
  }


  free_2d_array_complex(cumt1, ncsf); free_2d_array_complex(cumt2, ncsf);


  for (int i = 0; i < ncsf; i++) {
        free(hmt[i]);
  }
  free(hmt);

//-------------------------------------------------------------------------
  cumt1 = calloc_2d_array_complex(tchnl, tchnl);
  int findx;
  
  for(int i=0; i<tchnl; i++){
    findx = -(tchnl + 1)/2 + (i+1);
    com1 = -I*omga*0.5*dtim*findx;
    for(int j = 0; j<tchnl; j++){
      cumt1[i][j] = sqrt(2.0/(tchnl+1.0))*sin((i+1)*(j+1)*PI/(tchnl+1.0))*cexp(com1);
    }
  } 
  
// ================== Time Propagation starts =======================
  double complex *coff = (double complex*)calloc(nfloq, sizeof(double complex));
  double complex *cofh = (double complex*)calloc(ncsf, sizeof(double complex));
  double complex *cofc = (double complex*)calloc(ncsf, sizeof(double complex));
  
  double complex *h11, *h22, *h33;
  h11 = (double complex *)calloc(ncsf*ncsf, sizeof(double complex));
  h22 = (double complex *)calloc(ncsf, sizeof(double complex));
  h33 = (double complex *)calloc(ncsf, sizeof(double complex));

  com1 = -0.5*I*dtim;
  double time, epls, eff;
  int n1;
  double complex csum, dipole;


  FILE *timefile = fopen("./result/finalresult_serial.txt","w"); 


// ------------- time propagation loop starts ---------------- //

  for(int itim = 0; itim<ntim; itim++){
      startTime = omp_get_wtime(); 

      time = itim*dtim;
      epls = epsln*sin((PI*time)/totime)*sin((PI*time)/totime);
              
      n1 = 0;   
      for (int i=0; i<tchnl; i++){
        findx = -(tchnl + 2)/2 + i;

        for (int j=0; j<ncsf; j++){
            coff[n1] = cofi[inzr+j]*cexp(findx*I*omga*time);
            n1++;
        }
      }
      
// --------------------------------------------- 

      n1 = 0; eff=0.0;
      for (int i = 0; i < nfloq; i++) cofi[i] = CMPLX(0.0,0.0);

      for (int i=0; i<tchnl; i++){
        eff = epls*cos((i+1)*PI/(tchnl+1)); 
        for (int j=0; j<ncsf; j++){
            csum=CMPLX(0.0,0.0);  
            for (int k=0; k<tchnl; k++){
                csum = csum + cumt1[k][i]*coff[ncsf*k+j];
            }
            cofi[n1] = csum*cexp(com1*eff*zdipole[j]);
            n1++;
        }
      }

// ------------------ exhm ------------------

      n1 = 0; eff=0;
      for (int i = 0; i < nfloq; i++) coff[i] = CMPLX(0.0,0.0);

      for (int ii=0; ii<tchnl; ii++){
               
          int k = ii*ncsf;
               
          for (int j=0; j<ncsf; j++){
             h22[j] = cofi[k+j]; 
          }
          
          for(int i=0; i<ncsf; i++){
             for(int j=0; j<ncsf; j++){
            	     h11[j*ncsf+i] = exhm[i][j];
             }	  
          }
          
          for (int i = 0; i < ncsf; i++) h33[i] = CMPLX(0.0,0.0);

          cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                      ncsf, 1, ncsf, &alpha, h11, ncsf, 
                      h22, ncsf, &beta, h33, ncsf);
          
          for (int i=0; i<ncsf; i++){
             coff[k+i] = h33[i];
          } 

      }

// -------------------------------------

      n1 = 0; eff=0;
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
      
      for (int i = 0; i < ncsf; i++){
         cofh[i] = coff[inzr+i];
      }
    


      dipole = CMPLX(0.0,0.0);
      for (int i=0; i<ncsf; i++){
        dipole += conj(cofh[i])*epls*zdipole[i]*cos(omga*time)*cofh[i];
      }

      fprintf(timefile,"%e %e %e %e \n",time/tau, epls*cos(omga*time), creal(dipole), cimag(dipole)); 

      endTime = omp_get_wtime();

      serialTime = endTime - startTime;
      printf("Time taken in %d loop = %lf sec\n", itim, serialTime); 
  }   // time loop ends
  fclose(timefile);  


// free memory
  free(coff); free(cofi); free(cofh);
  free(h11); free(h22); free(h33);
  free(zdipole);

  free_2d_array_complex(cumt1, tchnl); 

  return 0;
}



int main(){
  int ntime, ncstep, istate, flqchnl;
  int ncsf, noptc;
  double xmin, xmax, epsln, omga;
  double tau, totime, lmda;
  double serialTime;


  ncsf = 1000; flqchnl = 3; xmin = 0 ; xmax = 10.0;
  epsln = 0.5; omga = 0.5; 
  noptc = 5; istate = 1;           //keep istate as 1, 0 giving wrong result
  
  tau = 2.0*PI/omga;
  totime = noptc*tau;
  ntime = (int)totime;

  double startTime = omp_get_wtime();
  timeprop(ncsf, xmin, xmax, flqchnl, noptc, istate, totime, ntime, omga, epsln);
  double endTime = omp_get_wtime();

  serialTime = endTime - startTime;
  printf("FINAL time taken by serial = %lf sec\n", serialTime);

  return 0;
}
