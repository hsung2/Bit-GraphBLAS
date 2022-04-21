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

// csc host
int *h_cscRowInd, *h_cscColPtr;

// csc device
int *cscRowInd, *cscColPtr;
float *cscVal;

// hsbsf metadata
int mb, nb, nblockrows;
int nblocks;

#if TILEDIM == 4
int tiledim = 4;
#elif TILEDIM == 8
int tiledim = 8;
#elif TILEDIM == 16
int tiledim = 16;
#elif TILEDIM == 32
int tiledim = 32;
#endif

// hsbsf
int *bsrRowPtr, *bsrColInd;

// hsbsf val
#if TILEDIM == 32
unsigned *tA;
#elif TILEDIM == 16
ushort *tA;
#else
uchar *tA;
#endif

// hsbsf vec
#if TILEDIM == 16
ushort *fC;
#elif TILEDIM == 32
unsigned *fC;
#else
uchar *fC;
#endif

// vec host
float *B;

// vec device
float *fB;

// pack B
#if TILEDIM == 32
unsigned *tB;
#elif TILEDIM == 16
ushort *tB;
#else
uchar *tB;
#endif

// cuSPARSE vec
float *dX, *dY;

// cuSPARSE result host
float *result_cusparsecsrspmvfloat;

// hsbsf result host
float *result_bsrbmv;

// hsbsf temp full vec
float *fC_full;

// cusparse handles
cusparseHandle_t handle_csr2csc;

cusparseMatDescr_t csr_descr = 0;
cusparseMatDescr_t bsr_descr = 0;
cudaStream_t streamId = 0;
cusparseHandle_t handle = 0;

cusparseHandle_t handle_csr;
cusparseMatDescr_t mat_A;
cusparseStatus_t cusparse_status;

/// ======================
void transposeMtx()
{
#ifdef VERIFY
    // csr2csc for B as A^T
    cudaMalloc(&cscRowInd, sizeof(int) * nnz);
    cudaMalloc(&cscColPtr, sizeof(int) * (nrows + 1));
    cudaMalloc(&cscVal, sizeof(float) * nnz);

    cusparseCreate(&handle_csr2csc);
    cusparseScsr2csc(handle_csr2csc, nrows, ncols, nnz,
                     csrVal, csrRowPtr, csrColInd,
                     cscVal, cscRowInd, cscColPtr,
                     CUSPARSE_ACTION_NUMERIC,
                     CUSPARSE_INDEX_BASE_ZERO);

    h_cscRowInd = (int *)malloc(sizeof(int) * nnz);
    h_cscColPtr = (int *)malloc(sizeof(int) * (nrows + 1));
    cudaMemcpy(h_cscRowInd, cscRowInd, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cscColPtr, cscColPtr, sizeof(int) * (nrows + 1), cudaMemcpyDeviceToHost);

    // change the csr indexing
    cudaMemcpy(csrRowPtr, cscColPtr, sizeof(int) * (nrows + 1), cudaMemcpyDeviceToDevice);
    cudaMemcpy(csrColInd, cscRowInd, sizeof(int) * nnz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(h_csrRowPtr, csrRowPtr, sizeof(int) * (nrows + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_csrColInd, csrColInd, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
#endif
}

void freeTransposeMtx()
{
    cusparseDestroy(handle_csr2csc);
    cudaFree(cscRowInd);
    cudaFree(cscColPtr);
    cudaFree(cscVal);
    free(h_cscRowInd);
    free(h_cscColPtr);
}

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

    // transpose mtx
    if (transpose)
    {
        // copy csc to csr with val remain the same
        transposeMtx();
        // free csc
        freeTransposeMtx();
    }
}

void CSR2HSBSF()
{
    // transform from csr to bsr using cuSPARSE API
    mb = (nrows + tiledim - 1) / tiledim;
    nb = (ncols + tiledim - 1) / tiledim;
    nblockrows = mb;

    // cuSPARSE API metadata setup

    cusparseCreateMatDescr(&csr_descr);
    cusparseSetMatType(csr_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(csr_descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseCreateMatDescr(&bsr_descr);
    cusparseSetMatType(bsr_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(bsr_descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseCreate(&handle);
    cusparseSetStream(handle, streamId);
    cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

    // csr2bsr in row-major order, estimate first
    cudaMalloc((void **)&bsrRowPtr, sizeof(int) * (nblockrows + 1));
    // GpuTimer csr_timer;
    // csr_timer.Start();
    cusparseXcsr2bsrNnz(handle, dirA, nrows, ncols, csr_descr,
                        csrRowPtr, csrColInd, tiledim, bsr_descr, bsrRowPtr, &nblocks);
    // csr_timer.Stop();
    // double cusparsecsrspmvfloat_time = csr_timer.ElapsedMillis();
    // printf("time: %f\n", cusparsecsrspmvfloat_time);
    cudaMalloc((void **)&bsrColInd, sizeof(int) * nblocks);

    // malloc packed matrix & pack
#if TILEDIM == 4
    cudaMalloc((void **)&tA, nblocks * tiledim * sizeof(uchar));
    csr2bsr_batch_4(h_csrRowPtr, h_csrColInd, nrows, ncols, nnz,
                    bsrRowPtr, bsrColInd, tA, tiledim, nblockrows, nblocks);
#elif TILEDIM == 8
    cudaMalloc((void **)&tA, nblocks * tiledim * sizeof(uchar));
    csr2bsr_batch_8(h_csrRowPtr, h_csrColInd, nrows, ncols, nnz,
                    bsrRowPtr, bsrColInd, tA, tiledim, nblockrows, nblocks);
#elif TILEDIM == 16
    cudaMalloc((void **)&tA, nblocks * tiledim * sizeof(ushort));
    csr2bsr_batch_16(h_csrRowPtr, h_csrColInd, nrows, ncols, nnz,
                     bsrRowPtr, bsrColInd, tA, tiledim, nblockrows, nblocks);
#elif TILEDIM == 32
    cudaMalloc((void **)&tA, nblocks * tiledim * sizeof(unsigned));
    csr2bsr_batch_32(h_csrRowPtr, h_csrColInd, nrows, ncols, nnz,
                     bsrRowPtr, bsrColInd, tA, tiledim, nblockrows, nblocks);
#endif
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

void packRandVec()
{
    // copy to device
    cudaMalloc(&fB, (nblockrows * tiledim) * sizeof(float));
    setDeviceValArr<int, float><<<1, 1>>>(fB, (nblockrows * tiledim), 0.0);
    cudaMemcpy(fB, B, nrows * sizeof(float), cudaMemcpyHostToDevice); // the rest are paddings

    // pack B
#if TILEDIM == 4
    int gridDim = (int)ceil(cbrt((double)nblockrows / 4));
    dim3 grid(gridDim, gridDim, gridDim);
    cudaMalloc(&tB, ceil((float)nblockrows / tiledim) * tiledim * sizeof(uchar));
    setDeviceValArr<int, uchar><<<1, 1>>>(tB, ceil((float)nblockrows / tiledim) * tiledim, 0);
    ToBit4Row<float><<<grid, 32>>>(fB, tB, nblockrows);
#elif TILEDIM == 8
    int gridDim = (int)ceil(cbrt((double)nblockrows / 4));
    dim3 grid(gridDim, gridDim, gridDim);
    cudaMalloc(&tB, ceil((float)nblockrows / tiledim) * tiledim * sizeof(uchar));
    setDeviceValArr<int, uchar><<<1, 1>>>(tB, ceil((float)nblockrows / tiledim) * tiledim, 0);
    ToBit8Row<float><<<grid, 32>>>(fB, tB, nblockrows);
#elif TILEDIM == 16
    int gridDim = (int)ceil(cbrt((double)nblockrows / 2));
    dim3 grid(gridDim, gridDim, gridDim);
    cudaMalloc(&tB, ceil((float)nblockrows / tiledim) * tiledim * sizeof(ushort));
    setDeviceValArr<int, ushort><<<1, 1>>>(tB, ceil((float)nblockrows / tiledim) * tiledim, 0);
    ToBit16Row<float><<<grid, 32>>>(fB, tB, nblockrows);
#elif TILEDIM == 32
    int gridDim = (int)ceil(cbrt((double)nblockrows));
    dim3 grid(gridDim, gridDim, gridDim);
    cudaMalloc(&tB, ceil((float)nblockrows / tiledim) * tiledim * sizeof(unsigned));
    setDeviceValArr<int, unsigned><<<1, 1>>>(tB, ceil((float)nblockrows / tiledim) * tiledim, 0);
    ToBit32Row<float><<<grid, 32>>>(fB, tB, nblockrows);
#endif
}

double evalCSRSpmvFloatCuSPARSE()
{
#ifdef VERIFY
    // metadata for cuSPARSE API
    cusparseCreate(&handle_csr);
    cusparseCreateMatDescr(&mat_A);
    cusparseSetMatType(mat_A, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(mat_A, CUSPARSE_INDEX_BASE_ZERO);

    // dummy multiplication variables
    // y = α ∗ op ( A ) ∗ x + β ∗ y
#if TEST_TIMES > 1
    float alpha = 1.0, beta = 1.0;
#else
    float alpha = 1.0, beta = 0.0;
#endif

    // create dense vector storage
    cudaMalloc((void **)&dX, sizeof(float) * nrows);
    cudaMemcpy(dX, B, sizeof(float) * nrows, cudaMemcpyHostToDevice); // do not move paddings
    cudaMalloc((void **)&dY, sizeof(float) * nrows);
    setDeviceValArr<int, float><<<1, 1>>>(dY, nrows, 0);

    // ------
    GpuTimer csr_timer;
    csr_timer.Start();

    for (int i = 0; i < TEST_TIMES; i++)
    {
        cusparseScsrmv(handle_csr, CUSPARSE_OPERATION_NON_TRANSPOSE, nrows, ncols, nnz,
                       &alpha, mat_A, csrVal, csrRowPtr, csrColInd, dX, &beta, dY);
    }

    csr_timer.Stop();
    double cusparsecsrspmvfloat_time = csr_timer.ElapsedMillis();
    // ------

    return cusparsecsrspmvfloat_time;
#endif
}

double evalHSBSFBmv()
{
    // init C (result storage)
#if TILEDIM == 4
    cudaMalloc(&fC, nblockrows * sizeof(uchar));
    setDeviceValArr<int, uchar><<<1, 1>>>(fC, nblockrows, 0);

    // int gridDim = (int)ceil(cbrt((double)nblockrows / (8*32)));
    int gridDim = (int)ceil(cbrt((double)nblockrows / 32));
    dim3 grid(gridDim, gridDim, gridDim);
#elif TILEDIM == 8
    cudaMalloc(&fC, nblockrows * sizeof(uchar));
    setDeviceValArr<int, uchar><<<1, 1>>>(fC, nblockrows, 0);

    // int gridDim = (int)ceil(cbrt((double)nblockrows / 4));
    int gridDim = (int)ceil(cbrt((double)nblockrows / (4*32)));
    dim3 grid(gridDim, gridDim, gridDim);
#elif TILEDIM == 16
    cudaMalloc(&fC, nblockrows * sizeof(ushort));
    setDeviceValArr<int, ushort><<<1, 1>>>(fC, nblockrows, 0);

    // int gridDim = (int)ceil(cbrt((double)nblockrows / 2));
    int gridDim = (int)ceil(cbrt((double)nblockrows / (2*32)));
    dim3 grid(gridDim, gridDim, gridDim);
#elif TILEDIM == 32
    cudaMalloc(&fC, nblockrows * sizeof(unsigned));
    setDeviceValArr<int, unsigned><<<1, 1>>>(fC, nblockrows, 0);

    int gridDim = (int)ceil(cbrt((double)nblockrows / 32));
    dim3 grid(gridDim, gridDim, gridDim);
#endif

    // ------
    GpuTimer bmv_timer;
    bmv_timer.Start();

    for (int i = 0; i < TEST_TIMES; i++)
    {
#if TILEDIM == 4
        bmv4_bin_bin_bin_new_1024<<<grid, 1024>>>(tA, tB, fC, bsrRowPtr, bsrColInd, nblockrows);
#elif TILEDIM == 8
        bmv8_bin_bin_bin_1024<<<grid, 1024>>>(tA, tB, fC, bsrRowPtr, bsrColInd, nblockrows);
#elif TILEDIM == 16
        bmv16_bin_bin_bin_1024<<<grid, 1024>>>(tA, tB, fC, bsrRowPtr, bsrColInd, nblockrows);
#elif TILEDIM == 32
        bmv32_bin_bin_bin<<<grid, 1024>>>(tA, tB, fC, bsrRowPtr, bsrColInd, nblockrows);
#endif
    }

    bmv_timer.Stop();
    double bmv_time = bmv_timer.ElapsedMillis() / double(TEST_TIMES);
    // ------

    return bmv_time;
}

/// ======================
void freeCSR()
{
    // free cusparse csr spmv
    cusparseDestroyMatDescr(mat_A);
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

void freeHSBSF()
{
    // free cusparse bsr metadata
    cusparseDestroyMatDescr(csr_descr);
    cusparseDestroyMatDescr(bsr_descr);
    cusparseDestroy(handle);

    // free storage
    cudaFree(tA);
    cudaFree(tB);

    // free vec mem
    free(B);
    cudaFree(fB);
    cudaFree(fC);

    // free indexing sys
    cudaFree(bsrRowPtr);
    cudaFree(bsrColInd);
}

void freeResult()
{
    free(result_bsrbmv);
    free(result_cusparsecsrspmvfloat);
    cudaFree(fC_full);
}

/// ======================
void verifyResult()
{
    // copy result to host for verification
    result_cusparsecsrspmvfloat = (float *)malloc(nrows * sizeof(float));
    cudaMemcpy(result_cusparsecsrspmvfloat, dY, nrows * sizeof(float), cudaMemcpyDeviceToHost);
    //printHostVec(result_cusparsecsrspmvfloat, nrows);

    // copy result to host for verification
    cudaMalloc((void **)&fC_full, sizeof(float) * tiledim * nblockrows);

#if TILEDIM == 4
    bin2full<uchar><<<1, 1>>>(fC, fC_full, nblockrows, 4);
#elif TILEDIM == 8
    bin2full<uchar><<<1, 1>>>(fC, fC_full, nblockrows, 8);
#elif TILEDIM == 16
    bin2full<ushort><<<1, 1>>>(fC, fC_full, nblockrows, 16);
#elif TILEDIM == 32
    bin2full<unsigned><<<1, 1>>>(fC, fC_full, nblockrows, 32);
#endif

#if TILEDIM == 4
//bin2full<uchar><<<(int)ceil(nblockrows / 1024.0), 1024>>>(fC, fC_full, nblockrows, 4);
#elif TILEDIM == 8
    printBin8Vec<<<1, 1>>>(fC, nblockrows);
#elif TILEDIM == 16
    printBin16Vec<<<1, 1>>>(fC, nblockrows);
#elif TILEDIM == 32
    printBin32Vec<<<1, 1>>>(fC, nblockrows);
#endif

    result_bsrbmv = (float *)malloc(nrows * sizeof(float)); // don't care padding result
    cudaMemcpy(result_bsrbmv, fC_full, nrows * sizeof(float), cudaMemcpyDeviceToHost);

    // verify bsrbmv with cuSPARSE baseline
    printf("success: %d\n", checkResultBin<float>(result_bsrbmv, result_cusparsecsrspmvfloat, nrows));

    // free mem
    freeResult();
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

#ifdef VERIFY
    // baseline
    double spmvtime = evalCSRSpmvFloatCuSPARSE();

    // hsbsf
    CSR2HSBSF();
    packRandVec();
    double bmvtime = evalHSBSFBmv();

    // verify result
    verifyResult();

    // free mem
    freeHSBSF();
    freeCSR();

    // print result
    printf("%f %f\n", spmvtime, bmvtime);

#else
    // hsbsf bmv
    CSR2HSBSF();
    packRandVec();
    double time = evalHSBSFBmv();

    // free mem
    freeHSBSF();
    freeCSR();

    // print result
    printf("%f ", time);
#endif
}
