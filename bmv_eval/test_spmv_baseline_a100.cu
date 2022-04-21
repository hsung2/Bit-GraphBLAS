/*
* cuSPARSE using CUDA 11.0 on A100 baseline
*/

#include <iostream>
#include <sys/time.h>

#define TEST_TIMES 5
using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

#include <vector>
#include "backend/readMtx.hpp"
#include "backend/csr2bsr_batch_bsrbmv.cu"

/// ======================
// csr metadata
int nrows, ncols, nnz;

// csr host
int *h_csrRowPtr, *h_csrColInd;
float *h_csrVal;

// csr device
int *csrRowPtr, *csrColInd;
float *csrVal;

// vec host
float *B;

// vec device
float *fB;

// cuSPARSE vec
float *dX, *dY;

// cusparse handles
cusparseHandle_t handle_csr2csc;

cusparseHandle_t handle_csr;
cusparseStatus_t cusparse_status;

cusparseSpMatDescr_t mat_A;
cusparseDnVecDescr_t vecX;
cusparseDnVecDescr_t vecY;

/// ======================
void readMtxCSR(const char *filename, bool transpose = false)
{
    // graphblast mmio interface
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<float> values;
    char *dat_name;
    readMtx(filename, &row_indices, &col_indices, &values,
            &nrows, &ncols, &nnz, 0, false, &dat_name); // directed, mtxinfo

    h_csrRowPtr = (int *)malloc(sizeof(int) * (nrows + 1));
    h_csrColInd = (int *)malloc(sizeof(int) * nnz);
    h_csrVal = (float *)malloc(sizeof(float) * nnz);
    coo2csr(h_csrRowPtr, h_csrColInd, h_csrVal,
            row_indices, col_indices, values, nrows, ncols);

    // copy csr to device
    cudaMalloc(&csrRowPtr, sizeof(int) * (nrows + 1));
    cudaMalloc(&csrColInd, sizeof(int) * nnz);
    cudaMalloc(&csrVal, sizeof(float) * nnz);
    cudaMemcpy(csrRowPtr, h_csrRowPtr, sizeof(int) * (nrows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(csrColInd, h_csrColInd, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(csrVal, h_csrVal, sizeof(float) * nnz, cudaMemcpyHostToDevice);

    // force all csrval to be 1 (this is for handling weighted adjacency matrix)
    setDeviceValArr<int, float><<<1, 1>>>(csrVal, nnz, 1.0);
}

void genRandVec()
{
    // generate random vector
    srand(time(0));
    B = (float *)malloc(nrows * sizeof(float));
    for (int i = 0; i < nrows; i++)
    {
        float x = (float)rand() / RAND_MAX;
        B[i] = (x > 0.5) ? 1 : 0;
    }
}

double evalCSRSpmvFloatCuSPARSE()
{
    // metadata for cuSPARSE API
    cusparseCreate(&handle_csr);

    // dummy multiplication variables
    // y = α ∗ op ( A ) ∗ x + β ∗ y
#if TEST_TIMES > 1
    float alpha = 1.0, beta = 1.0;
#else
    float alpha = 1.0, beta = 0.0;
#endif

    // create CSR
    cusparseCreateCsr(&mat_A, nrows, ncols, nnz,
                      csrRowPtr, csrColInd, csrVal,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // create dense vector storage
    cudaMalloc((void **)&dX, sizeof(float) * nrows);
    cudaMemcpy(dX, B, sizeof(float) * nrows, cudaMemcpyHostToDevice); // do not move paddings
    cudaMalloc((void **)&dY, sizeof(float) * nrows);
    setDeviceValArr<int, float><<<1, 1>>>(dY, nrows, 0);

    cusparseCreateDnVec(&vecX, nrows, dX, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, nrows, dY, CUDA_R_32F);

    // buffer
    void *buffer;
    size_t tempInt;
    cusparseSpMV_bufferSize(handle_csr, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, mat_A, vecX, &beta, vecY,
                            CUDA_R_32F, CUSPARSE_CSRMV_ALG1, &tempInt);
    cudaMalloc(&buffer, tempInt);

    // ------
    GpuTimer csr_timer;
    csr_timer.Start();

    for (int i = 0; i < TEST_TIMES; i++)
    {
        cusparseSpMV(handle_csr, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_A, vecX, &beta, vecY,
                     CUDA_R_32F, CUSPARSE_CSRMV_ALG1, &buffer);
    }

    csr_timer.Stop();
    double cusparsecsrspmvfloat_time = csr_timer.ElapsedMillis() / double(TEST_TIMES);
    // ------

    // free temp storage
    cudaFree(buffer);

    return cusparsecsrspmvfloat_time;
}

void freeCSR()
{
    // free cusparse csr spmv
    cusparseDestroySpMat(mat_A);
    cusparseDestroy(handle_csr);

    // free csr mem
    free(h_csrRowPtr);
    free(h_csrColInd);
    free(h_csrVal);

    cudaFree(csrRowPtr);
    cudaFree(csrColInd);
    cudaFree(csrVal);

    // free vec mem
    cudaFree(dX);
    cudaFree(dY);
}

/// ======================
void verifyResult()
{
    // copy result to host for verification
    float *Y;
    cudaMalloc((void **)&Y, sizeof(float) * nrows);
    float *result_cusparsecsrspmvfloat = (float *)malloc(sizeof(float) * nrows);
    cusparseDnVecGetValues(vecY, (void **)&Y);
    cudaMemcpy(result_cusparsecsrspmvfloat, Y, sizeof(float) * nrows, cudaMemcpyDeviceToHost);
    cudaFree(Y);

    // verify bsrbmv with cuSPARSE baseline
    for (int i = 0; i < nrows; i++)
        printf("%f ", result_cusparsecsrspmvfloat[i]);
    printf("\n");

    // free mem
    free(result_cusparsecsrspmvfloat);
}

/// ======================
int main(int argc, char *argv[])
{
    char *filename = argv[1];      // e.g. "G43.mtx"
    int transpose = atoi(argv[2]); // 1: transpose, 0: default

    // bmv: C = A * B
    // init
    cudaSetDevice(0);
    readMtxCSR(filename, transpose);
    genRandVec();

    // csr spmv
    double time = evalCSRSpmvFloatCuSPARSE();

    // verify
    // verifyResult();

    // free mem
    freeCSR();

    // print result
    printf("%f ", time);
}
