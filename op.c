#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<complex.h>
#include<lapacke.h>
#include<cblas.h>
#include<omp.h>

#define PI 3.1415926535897932

double potential(double x) {
  double pot;
  pot = 0.50*x*x;  
  return pot;
}

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

int timeprop(int ncsf, double xmin, double xmax, double lmda, int flqchnl, int noptc, 
             int istate, double totime, int ntim, double omga, double epsln) {

  double mass = 10000.0; 
  double dtim = totime/ntim;
  double dx = (xmax-xmin)/(ncsf-1);
  int tchnl = 2*flqchnl+1;
  int nfloq = ncsf*tchnl;
  int inzr = flqchnl*ncsf;
  double tau = 2.0*PI/omga;

  double **hmt = (double**)calloc(ncsf,sizeof(double*));
  
  for (int i=0; i<ncsf; i++){
    hmt[i] = (double*)calloc(ncsf,sizeof(double));
  }

  for(int k=0; k<ncsf; k++){
    hmt[k][k] = (PI*PI)/(6.0*dx*dx);
    for(int j=0; j<k; j++){	    
      hmt[k][j] = (pow(-1.0,(k-j)))/((k-j)*(k-j)*dx*dx);
      hmt[j][k] = hmt[k][j];
    }
  }

  FILE *file = fopen("./result/kinetic.txt","w");  
  for(int i=0; i<ncsf; i++){
    for(int j=0; j<ncsf; j++){
	fprintf(file,"%e ",hmt[i][j]);
    }	  
    fprintf(file,"\n");
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

  double *eig = (double *)calloc(ncsf,sizeof(double));  
  int info;
  info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', ncsf, h, ncsf, eig);

  if (info != 0) {
      printf("Eror in LAPACKE_dsyev: %d\n",info);
      return -1;
  }

  for(int i=0; i<ncsf; i++){
    for(int j=0; j<ncsf; j++){
      hmt[i][j] = h[i*ncsf+j];
    }
  }

  double complex *cofi = (double complex*)calloc(nfloq, sizeof(double complex));

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
    	h1[j*ncsf+i] = cumt1[i][j];        
      h2[j*ncsf+i] = cumt2[i][j];        
    }	  
  }

  double complex alpha = 1.0 + 0.0*I;
  double complex beta = 0.0 + 0.0*I;

  cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              ncsf, ncsf, ncsf, &alpha, h2, ncsf, 
              h1, ncsf, &beta, h3, ncsf);

  for(int i=0; i<ncsf; i++){
    for(int j=0; j<ncsf; j++){
      exhm[i][j] = h3[j*ncsf+i];
    }
  }

  for (int i = 0; i < ncsf; i++){
    for (int j=0; j<ncsf; j++){
        cumt1[i][j] = CMPLX(0.0,0.0); 
    }
  } 

  for(int i=0; i<ncsf; i++){
    cumt1[i][i] = cexp(-lmda*cpm[i]*dtim);
  }

  free(cpm);

  copy_matrix_complex_to_complex(ncsf, exhm, cumt2);

  for(int i=0; i<ncsf; i++){
    for(int j=0; j<ncsf; j++){
    	h1[j*ncsf+i] = cumt1[i][j];
      h2[j*ncsf+i] = cumt2[i][j];
    }	  
  }

  cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              ncsf, ncsf, ncsf, &alpha, h2, ncsf, 
              h1, ncsf, &beta, h3, ncsf);

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

  cumt1 = calloc_2d_array_complex(tchnl, tchnl);
  int findx;
  
  for(int i=0; i<tchnl; i++){
    findx = -(tchnl + 1)/2 + (i+1);
    com1 = -I*omga*0.5*dtim*findx;
    for(int j = 0; j<tchnl; j++){
      cumt1[i][j] = sqrt(2.0/(tchnl+1.0))*sin((i+1)*(j+1)*PI/(tchnl+1.0))*cexp(com1);
    }
  } 

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

  FILE *timefile = fopen("./result/finalresult.txt","w"); 

  #pragma omp parallel for private(time, epls, eff, n1, csum, dipole)
  for(int itim = 0; itim<ntim; itim++){
       
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
      
      for (int i = 0; i< nfloq; i++){
        fprintf(timefile,"%e ",creal(cofi[i]));
      }
      fprintf(timefile,"\n");
  }
  
  fclose(timefile);
  
  free(coff);
  free(cofi);
  free(cofh);
  return 0;
}

int main() {
    int ntime, ncstep, istate, flqchnl;
    int ncsf, noptc;
    double xmin, xmax, epsln, omga;
    double tau, totime, lmda;

    ncsf = 1000; flqchnl = 5; xmin = 0; xmax = 10.0;
    epsln = 0.5; omga = 0.5; lmda = 0.0;
    noptc = 5; istate = 1;

    tau = 2.0 * PI / omga;
    totime = noptc * tau;
    ntime = (int)totime;

    printf("ntime = %d, totime = %lf, tau = %lf\n", ntime, totime, tau);

  float start=omp_get_wtime();
    timeprop(ncsf, xmin, xmax, lmda, flqchnl, noptc, istate, totime, ntime, omga, epsln);
float stop=omp_get_wtime();
printf("t time %f",stop-start);
    return 0;
}

