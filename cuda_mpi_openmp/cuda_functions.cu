#include<stdio.h>
#include<stdlib.h>

#include<cuda_runtime.h>
#include<cusolverDn.h>

#include<cublas_v2.h>
#include<cuComplex.h>


void cuda_diagonalize_matrix(double *h_A, int N, double *h_evec, double *h_eval)
{
    double *d_A, *d_W; // Device memory for matrix and eigenvalues
    cudaError_t cudaStat;
    cusolverStatus_t cusolverStat;
    
    // Allocate memory on the device
    cudaStat = cudaMalloc((void**)&d_A, N * N * sizeof(double));
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed!\n");
        return;
    }
    
    cudaStat = cudaMalloc((void**)&d_W, N * sizeof(double));
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed!\n");
        cudaFree(d_A);
        return;
    }

    // Copy matrix A to device
    cudaStat = cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy failed!\n");
        cudaFree(d_A);
        cudaFree(d_W);
        return;
    }

    // Create cuSOLVER handle
    cusolverDnHandle_t cusolverH;
    cusolverStat = cusolverDnCreate(&cusolverH);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "cuSOLVER handle creation failed!\n");
        cudaFree(d_A);
        cudaFree(d_W);
        return;
    }

    // Query buffer size
    int Lwork;
    cusolverStat = cusolverDnDsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
                                               CUBLAS_FILL_MODE_LOWER, N, d_A, N, d_W, &Lwork);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "Buffer size query failed!\n");
        cusolverDnDestroy(cusolverH);
        cudaFree(d_A);
        cudaFree(d_W);
        return;
    }

    // Allocate workspace
    double *d_work;
    cudaStat = cudaMalloc((void**)&d_work, Lwork * sizeof(double));
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "CUDA malloc for workspace failed!\n");
        cusolverDnDestroy(cusolverH);
        cudaFree(d_A);
        cudaFree(d_W);
        return;
    }

    int *d_info;
    cudaStat = cudaMalloc((void**)&d_info, sizeof(int));
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "CUDA malloc for info failed!\n");
        cudaFree(d_work);
        cusolverDnDestroy(cusolverH);
        cudaFree(d_A);
        cudaFree(d_W);
        return;
    }

    // Compute eigenvalues and eigenvectors
    cusolverStat = cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                                    N, d_A, N, d_W, d_work, Lwork, d_info);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "Eigen decomposition failed!\n");
        cudaFree(d_info);
        cudaFree(d_work);
        cusolverDnDestroy(cusolverH);
        cudaFree(d_A);
        cudaFree(d_W);
        return;
    }

    // Copy results back to host

    // Copy back eigenvalues
    cudaStat = cudaMemcpy(h_eval, d_W, N * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy for eigenvalues failed!\n");
    }

    // Copy back eigenvectors
    cudaStat = cudaMemcpy(h_evec, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy for eigenvectors failed!\n");
    }

    int info;
    cudaStat = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy for info failed!\n");
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_W);
    cudaFree(d_work);
    cudaFree(d_info);
    cusolverDnDestroy(cusolverH);
}


void cuda_complexMatrixMultiply(const cuDoubleComplex *A, const cuDoubleComplex *B, cuDoubleComplex *C, int n) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cuDoubleComplex *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, n * n * sizeof(cuDoubleComplex));
    cudaMalloc((void **)&d_B, n * n * sizeof(cuDoubleComplex));
    cudaMalloc((void **)&d_C, n * n * sizeof(cuDoubleComplex));
    
    cublasSetMatrix(n, n, sizeof(cuDoubleComplex), A, n, d_A, n);
    cublasSetMatrix(n, n, sizeof(cuDoubleComplex), B, n, d_B, n);
    
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n);
    
    cublasGetMatrix(n, n, sizeof(cuDoubleComplex), d_C, n, C, n);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}

void cuda_complexMatrixMultiply_col(const cuDoubleComplex *A, const cuDoubleComplex *B, cuDoubleComplex *C, int n) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cuDoubleComplex *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, n * n * sizeof(cuDoubleComplex));
    cudaMalloc((void **)&d_B, n * 1 * sizeof(cuDoubleComplex));
    cudaMalloc((void **)&d_C, n * 1 * sizeof(cuDoubleComplex));
    
    cublasSetMatrix(n, n, sizeof(cuDoubleComplex), A, n, d_A, n);
    cublasSetMatrix(n, 1, sizeof(cuDoubleComplex), B, n, d_B, n);
    
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

    cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, 1, n, &alpha, d_A, n, d_B, n, &beta, d_C, n);
    
    cublasGetMatrix(n, 1, sizeof(cuDoubleComplex), d_C, n, C, n);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}