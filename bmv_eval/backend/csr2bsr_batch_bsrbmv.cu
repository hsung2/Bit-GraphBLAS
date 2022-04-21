/**
* Filename: csr2bsr_batch_bsrbmv.cu
*
* Description: batch processing functions for csr to b2sr conversion.
*
*/

#include "bsrbmv.cu"
#include "utility.cu"

/**
* batch the process of csr2bsr, blocksize = 4
* assume csr val are only 0 or 1
* each value store at the lower 4 bit
*/
void csr2bsr_batch_4(const int *h_csrRowPtr, const int *h_csrColInd,
                     const int nrows, const int ncols, const int nnz,
                     int *bsrRowPtr, int *bsrColInd, uchar *bsrVal,
                     const int blocksize, const int nblockrows, const int nblocks)
{
    // global result
    cudaMemset(bsrColInd, 0, nblocks);
    cudaMemset(bsrVal, 0, nblocks * blocksize);
    int total_nblocks = 0;

    // buffer storage
    int *temp_csrrowptr, *temp_csrcolind;
    float *temp_csrval;
    int *temp_bsrrowptr, *temp_bsrcolind;
    float *temp_bsrval;
    uchar *temp_bsrval_packed;

// #pragma omp parallel for
    for (int i = 0; i < nblockrows; i++)
    { // serial processing, 1 blockrow at a time

        // calculate nnz in this blockrow
        int temp_rowend = ((i + 1) * 4 < nrows ? h_csrRowPtr[(i + 1) * 4] : nnz), temp_rowstart = h_csrRowPtr[i * 4];
        int temp_nnz = temp_rowend - temp_rowstart;

        if (temp_nnz != 0)
        {
            // set up cusparse metadata
            // 1) create cusparsematdescr for csr, bsr
            cusparseMatDescr_t csr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&csr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(csr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(csr_descr, CUSPARSE_INDEX_BASE_ZERO));
            cusparseMatDescr_t bsr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&bsr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(bsr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(bsr_descr, CUSPARSE_INDEX_BASE_ZERO));

            // 2) cusparse handle, stream
            cudaStream_t streamId = 0;
            cusparseHandle_t handle = 0;
            CHECK_CUSPARSE(cusparseCreate(&handle));
            CHECK_CUSPARSE(cusparseSetStream(handle, streamId));

            // 3) cusparse direction
            cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

            /* **************************************** */
            // set buffer csr storage
            // 1) set buffer csr rowptr & offset
            cudaMalloc((void **)&temp_csrrowptr, sizeof(int) * (4 + 1));
            if (i == nblockrows - 1)
            { // last iteration
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 4, sizeof(int) * ((nrows + 1) - (i * 4)), cudaMemcpyHostToDevice);
                offsetDeviceArr<int><<<(((nrows + 1) - (i * 4))+1024-1)/1024,1024>>>(temp_csrrowptr, ((nrows + 1) - (i * 4)), temp_rowstart); // offset rowptr
                padDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, ((nrows + 1) - (i * 4)), (4 + 1), temp_nnz);
            }
            else
            { // all other iteration except last
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 4, sizeof(int) * (4 + 1), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, (4 + 1), temp_rowstart); // offset rowptr
            }

            // 2) set buffer csr colind
            cudaMalloc((void **)&temp_csrcolind, sizeof(int) * temp_nnz);
            cudaMemcpy(temp_csrcolind, h_csrColInd + temp_rowstart, sizeof(int) * temp_nnz, cudaMemcpyHostToDevice);

            // 3) set buffer csr val
            cudaMalloc((void **)&temp_csrval, sizeof(float) * temp_nnz);
            setDeviceArr<float><<<(temp_nnz+1024-1)/1024, 1024>>>(temp_csrval, temp_nnz, 1.0);

            // calculate nnzb & allocate buffer bsr storage
            int h_bsrRowPtr_0, h_bsrRowPtr_1;
            cudaMemcpy(&h_bsrRowPtr_0, bsrRowPtr+i, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_bsrRowPtr_1, bsrRowPtr+i+1, sizeof(int), cudaMemcpyDeviceToHost);
            int temp_nblocks = h_bsrRowPtr_1 - h_bsrRowPtr_0;
            cudaMalloc((void **)&temp_bsrrowptr, sizeof(int) * 2);
            setDeviceIndArrElem<int><<<1, 1>>>(temp_bsrrowptr, 0, 0);
            setDeviceIndArrElem<int><<<1, 1>>>(temp_bsrrowptr, 1, temp_nblocks);

            cudaMalloc((void **)&temp_bsrcolind, sizeof(int) * temp_nblocks);
            cudaMalloc((void **)&temp_bsrval, sizeof(float) * temp_nblocks * blocksize * blocksize);

            // csr2bsr (nrows = blocksize, ncols = ncols)
            CHECK_CUSPARSE(cusparseScsr2bsr(handle, dirA, blocksize, ncols, csr_descr,
                                            temp_csrval, temp_csrrowptr, temp_csrcolind,
                                            blocksize, bsr_descr, temp_bsrval, temp_bsrrowptr, temp_bsrcolind));
            total_nblocks += temp_nblocks;

            // pack buffer bsrval to binary
            cudaMalloc((void **)&temp_bsrval_packed, sizeof(uchar) * ceil((float)temp_nblocks / 64) * 64 * blocksize);
            ToBit4Col<float><<<dim3(1, ceil((float)temp_nblocks / 64)), 32>>>(temp_bsrval, temp_bsrval_packed, temp_nblocks);

            cudaMemcpy(bsrColInd + h_bsrRowPtr_0, temp_bsrcolind, sizeof(int) * temp_nblocks, cudaMemcpyDeviceToDevice);
            cudaMemcpy(bsrVal + h_bsrRowPtr_0 * blocksize, temp_bsrval_packed, sizeof(uchar) * temp_nblocks * blocksize, cudaMemcpyDeviceToDevice);

            // clean buffer
            cudaFree(temp_csrrowptr);
            temp_csrrowptr = NULL;
            cudaFree(temp_csrcolind);
            temp_csrcolind = NULL;
            cudaFree(temp_csrval);
            temp_csrval = NULL;
            cudaFree(temp_bsrrowptr);
            temp_bsrrowptr = NULL;
            cudaFree(temp_bsrcolind);
            temp_bsrcolind = NULL;
            cudaFree(temp_bsrval);
            temp_bsrval = NULL;
            cudaFree(temp_bsrval_packed);
            temp_bsrval_packed = NULL;

            // free descr and handle memory
            CHECK_CUSPARSE(cusparseDestroyMatDescr(csr_descr));
            CHECK_CUSPARSE(cusparseDestroyMatDescr(bsr_descr));
            CHECK_CUSPARSE(cusparseDestroy(handle));
        }
    }
}

/**
* batch the process of csr2bsr, blocksize = 8
* assume csr val are only 0 or 1
*/
void csr2bsr_batch_8(const int *h_csrRowPtr, const int *h_csrColInd,
                     const int nrows, const int ncols, const int nnz,
                     int *bsrRowPtr, int *bsrColInd, uchar *bsrVal,
                     const int blocksize, const int nblockrows, const int nblocks)
{
    // global result
    cudaMemset(bsrColInd, 0, nblocks);
    cudaMemset(bsrVal, 0, nblocks * blocksize);
    int total_nblocks = 0;

    // buffer storage
    int *temp_csrrowptr, *temp_csrcolind;
    float *temp_csrval;
    int *temp_bsrrowptr, *temp_bsrcolind;
    float *temp_bsrval;
    uchar *temp_bsrval_packed;

    for (int i = 0; i < nblockrows; i++)
    { // serial processing, 1 blockrow at a time

        // calculate nnz in this blockrow
        int temp_rowend = ((i + 1) * 8 < nrows ? h_csrRowPtr[(i + 1) * 8] : nnz), temp_rowstart = h_csrRowPtr[i * 8];
        int temp_nnz = temp_rowend - temp_rowstart;

        if (temp_nnz != 0)
        {
            // set up cusparse metadata
            // 1) create cusparsematdescr for csr, bsr
            cusparseMatDescr_t csr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&csr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(csr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(csr_descr, CUSPARSE_INDEX_BASE_ZERO));
            cusparseMatDescr_t bsr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&bsr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(bsr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(bsr_descr, CUSPARSE_INDEX_BASE_ZERO));

            // 2) cusparse handle, stream
            cudaStream_t streamId = 0;
            cusparseHandle_t handle = 0;
            CHECK_CUSPARSE(cusparseCreate(&handle));
            CHECK_CUSPARSE(cusparseSetStream(handle, streamId));

            // 3) cusparse direction
            cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

            /* **************************************** */
            // set buffer csr storage
            // 1) set buffer csr rowptr & offset
            cudaMalloc((void **)&temp_csrrowptr, sizeof(int) * (8 + 1));
            if (i == nblockrows - 1)
            { // last iteration
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 8, sizeof(int) * ((nrows + 1) - (i * 8)), cudaMemcpyHostToDevice);
                offsetDeviceArr<int><<<(((nrows + 1) - (i * 8))+1024-1)/1024,1024>>>(temp_csrrowptr, ((nrows + 1) - (i * 8)), temp_rowstart); // offset rowptr
                padDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, ((nrows + 1) - (i * 8)), (8 + 1), temp_nnz);
            }
            else
            { // all other iteration except last
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 8, sizeof(int) * (8 + 1), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, (8 + 1), temp_rowstart); // offset rowptr
            }

            // 2) set buffer csr colind
            cudaMalloc((void **)&temp_csrcolind, sizeof(int) * temp_nnz);
            cudaMemcpy(temp_csrcolind, h_csrColInd + temp_rowstart, sizeof(int) * temp_nnz, cudaMemcpyHostToDevice);

            // 3) set buffer csr val
            cudaMalloc((void **)&temp_csrval, sizeof(float) * temp_nnz);
            setDeviceArr<float><<<(temp_nnz+1024-1)/1024, 1024>>>(temp_csrval, temp_nnz, 1.0);

            // calculate nnzb & allocate buffer bsr storage
            int h_bsrRowPtr_0, h_bsrRowPtr_1;
            cudaMemcpy(&h_bsrRowPtr_0, bsrRowPtr+i, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_bsrRowPtr_1, bsrRowPtr+i+1, sizeof(int), cudaMemcpyDeviceToHost);
            int temp_nblocks = h_bsrRowPtr_1 - h_bsrRowPtr_0;
            cudaMalloc((void **)&temp_bsrrowptr, sizeof(int) * 2);
            setDeviceIndArrElem<int><<<1, 1>>>(temp_bsrrowptr, 0, 0);
            setDeviceIndArrElem<int><<<1, 1>>>(temp_bsrrowptr, 1, temp_nblocks);

            cudaMalloc((void **)&temp_bsrcolind, sizeof(int) * temp_nblocks);
            cudaMalloc((void **)&temp_bsrval, sizeof(float) * temp_nblocks * blocksize * blocksize);

            // csr2bsr (nrows = blocksize, ncols = ncols)
            CHECK_CUSPARSE(cusparseScsr2bsr(handle, dirA, blocksize, ncols, csr_descr,
                                            temp_csrval, temp_csrrowptr, temp_csrcolind,
                                            blocksize, bsr_descr, temp_bsrval, temp_bsrrowptr, temp_bsrcolind));
            total_nblocks += temp_nblocks;

            // pack buffer bsrval to binary
            cudaMalloc((void **)&temp_bsrval_packed, sizeof(uchar) * ceil((float)temp_nblocks / 16) * 16 * blocksize);
            ToBit8Col<float><<<dim3(1, ceil((float)temp_nblocks / 16)), 32>>>(temp_bsrval, temp_bsrval_packed, temp_nblocks);

            cudaMemcpy(bsrColInd + h_bsrRowPtr_0, temp_bsrcolind, sizeof(int) * temp_nblocks, cudaMemcpyDeviceToDevice);
            cudaMemcpy(bsrVal + h_bsrRowPtr_0 * blocksize, temp_bsrval_packed, sizeof(uchar) * temp_nblocks * blocksize, cudaMemcpyDeviceToDevice);

            // clean buffer
            cudaFree(temp_csrrowptr);
            temp_csrrowptr = NULL;
            cudaFree(temp_csrcolind);
            temp_csrcolind = NULL;
            cudaFree(temp_csrval);
            temp_csrval = NULL;
            cudaFree(temp_bsrrowptr);
            temp_bsrrowptr = NULL;
            cudaFree(temp_bsrcolind);
            temp_bsrcolind = NULL;
            cudaFree(temp_bsrval);
            temp_bsrval = NULL;
            cudaFree(temp_bsrval_packed);
            temp_bsrval_packed = NULL;

            // free descr and handle memory
            CHECK_CUSPARSE(cusparseDestroyMatDescr(csr_descr));
            CHECK_CUSPARSE(cusparseDestroyMatDescr(bsr_descr));
            CHECK_CUSPARSE(cusparseDestroy(handle));
        }
    }
}

/**
* batch the process of csr2bsr, blocksize = 16
* assume csr val are only 0 or 1
*/
void csr2bsr_batch_16(const int *h_csrRowPtr, const int *h_csrColInd,
                      const int nrows, const int ncols, const int nnz,
                      int *bsrRowPtr, int *bsrColInd, ushort *bsrVal,
                      const int blocksize, const int nblockrows, const int nblocks)
{
    // global result
    cudaMemset(bsrColInd, 0, nblocks);
    cudaMemset(bsrVal, 0, nblocks * blocksize);
    int total_nblocks = 0;

    // buffer storage
    int *temp_csrrowptr, *temp_csrcolind;
    float *temp_csrval;
    int *temp_bsrrowptr, *temp_bsrcolind;
    float *temp_bsrval;
    ushort *temp_bsrval_packed;

    for (int i = 0; i < nblockrows; i++)
    { // serial processing, 1 blockrow at a time

        // calculate nnz in this blockrow
        int temp_rowend = ((i + 1) * 16 < nrows ? h_csrRowPtr[(i + 1) * 16] : nnz), temp_rowstart = h_csrRowPtr[i * 16];
        int temp_nnz = temp_rowend - temp_rowstart;

        if (temp_nnz != 0)
        {
            // set up cusparse metadata
            // 1) create cusparsematdescr for csr, bsr
            cusparseMatDescr_t csr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&csr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(csr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(csr_descr, CUSPARSE_INDEX_BASE_ZERO));
            cusparseMatDescr_t bsr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&bsr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(bsr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(bsr_descr, CUSPARSE_INDEX_BASE_ZERO));

            // 2) cusparse handle, stream
            cudaStream_t streamId = 0;
            cusparseHandle_t handle = 0;
            CHECK_CUSPARSE(cusparseCreate(&handle));
            CHECK_CUSPARSE(cusparseSetStream(handle, streamId));

            // 3) cusparse direction
            cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

            /* **************************************** */
            // set buffer csr storage
            // 1) set buffer csr rowptr & offset
            cudaMalloc((void **)&temp_csrrowptr, sizeof(int) * (16 + 1));
            if (i == nblockrows - 1)
            { // last iteration
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 16, sizeof(int) * ((nrows + 1) - (i * 16)), cudaMemcpyHostToDevice);
                offsetDeviceArr<int><<<(((nrows + 1) - (i * 16))+1024-1)/1024,1024>>>(temp_csrrowptr, ((nrows + 1) - (i * 16)), temp_rowstart); // offset rowptr
                padDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, ((nrows + 1) - (i * 16)), (16 + 1), temp_nnz);
            }
            else
            { // all other iteration except last
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 16, sizeof(int) * (16 + 1), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, (16 + 1), temp_rowstart); // offset rowptr
            }

            // 2) set buffer csr colind
            cudaMalloc((void **)&temp_csrcolind, sizeof(int) * temp_nnz);
            cudaMemcpy(temp_csrcolind, h_csrColInd + temp_rowstart, sizeof(int) * temp_nnz, cudaMemcpyHostToDevice);

            // 3) set buffer csr val
            cudaMalloc((void **)&temp_csrval, sizeof(float) * temp_nnz);
            setDeviceArr<float><<<(temp_nnz+1024-1)/1024, 1024>>>(temp_csrval, temp_nnz, 1.0);

            // calculate nnzb & allocate buffer bsr storage
            int h_bsrRowPtr_0, h_bsrRowPtr_1;
            cudaMemcpy(&h_bsrRowPtr_0, bsrRowPtr+i, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_bsrRowPtr_1, bsrRowPtr+i+1, sizeof(int), cudaMemcpyDeviceToHost);
            int temp_nblocks = h_bsrRowPtr_1 - h_bsrRowPtr_0;
            cudaMalloc((void **)&temp_bsrrowptr, sizeof(int) * 2);
            setDeviceIndArrElem<int><<<1, 1>>>(temp_bsrrowptr, 0, 0);
            setDeviceIndArrElem<int><<<1, 1>>>(temp_bsrrowptr, 1, temp_nblocks);

            cudaMalloc((void **)&temp_bsrcolind, sizeof(int) * temp_nblocks);
            cudaMalloc((void **)&temp_bsrval, sizeof(float) * temp_nblocks * blocksize * blocksize);

            // csr2bsr (nrows = blocksize, ncols = ncols)
            CHECK_CUSPARSE(cusparseScsr2bsr(handle, dirA, blocksize, ncols, csr_descr,
                                            temp_csrval, temp_csrrowptr, temp_csrcolind,
                                            blocksize, bsr_descr, temp_bsrval, temp_bsrrowptr, temp_bsrcolind));
            total_nblocks += temp_nblocks;

            // pack buffer bsrval to binary
            cudaMalloc((void **)&temp_bsrval_packed, sizeof(ushort) * ceil((float)temp_nblocks / 4) * 4 * blocksize);
            ToBit16Col<float><<<dim3(1, ceil((float)temp_nblocks / 4)), 32>>>(temp_bsrval, temp_bsrval_packed, temp_nblocks);

            cudaMemcpy(bsrColInd + h_bsrRowPtr_0, temp_bsrcolind, sizeof(int) * temp_nblocks, cudaMemcpyDeviceToDevice);
            cudaMemcpy(bsrVal + h_bsrRowPtr_0 * blocksize, temp_bsrval_packed, sizeof(uchar) * temp_nblocks * blocksize, cudaMemcpyDeviceToDevice);

            // clean buffer
            cudaFree(temp_csrrowptr);
            temp_csrrowptr = NULL;
            cudaFree(temp_csrcolind);
            temp_csrcolind = NULL;
            cudaFree(temp_csrval);
            temp_csrval = NULL;
            cudaFree(temp_bsrrowptr);
            temp_bsrrowptr = NULL;
            cudaFree(temp_bsrcolind);
            temp_bsrcolind = NULL;
            cudaFree(temp_bsrval);
            temp_bsrval = NULL;
            cudaFree(temp_bsrval_packed);
            temp_bsrval_packed = NULL;

            // free descr and handle memory
            CHECK_CUSPARSE(cusparseDestroyMatDescr(csr_descr));
            CHECK_CUSPARSE(cusparseDestroyMatDescr(bsr_descr));
            CHECK_CUSPARSE(cusparseDestroy(handle));
        }
    }
}

/**
* batch the process of csr2bsr, blocksize = 32
* assume csr val are only 0 or 1
*/
void csr2bsr_batch_32(const int *h_csrRowPtr, const int *h_csrColInd,
                      const int nrows, const int ncols, const int nnz,
                      int *bsrRowPtr, int *bsrColInd, unsigned *bsrVal,
                      const int blocksize, const int nblockrows, const int nblocks)
{
    // global result
    cudaMemset(bsrColInd, 0, nblocks);
    cudaMemset(bsrVal, 0, nblocks * blocksize);
    int total_nblocks = 0;

    // buffer storage
    int *temp_csrrowptr, *temp_csrcolind;
    float *temp_csrval;
    int *temp_bsrrowptr, *temp_bsrcolind;
    float *temp_bsrval;
    unsigned *temp_bsrval_packed;

    for (int i = 0; i < nblockrows; i++)
    { // serial processing, 1 blockrow at a time

        // calculate nnz in this blockrow
        int temp_rowend = ((i + 1) * 32 < nrows ? h_csrRowPtr[(i + 1) * 32] : nnz), temp_rowstart = h_csrRowPtr[i * 32];
        int temp_nnz = temp_rowend - temp_rowstart;

        if (temp_nnz != 0)
        {
            // set up cusparse metadata
            // 1) create cusparsematdescr for csr, bsr
            cusparseMatDescr_t csr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&csr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(csr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(csr_descr, CUSPARSE_INDEX_BASE_ZERO));
            cusparseMatDescr_t bsr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&bsr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(bsr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(bsr_descr, CUSPARSE_INDEX_BASE_ZERO));

            // 2) cusparse handle, stream
            cudaStream_t streamId = 0;
            cusparseHandle_t handle = 0;
            CHECK_CUSPARSE(cusparseCreate(&handle));
            CHECK_CUSPARSE(cusparseSetStream(handle, streamId));

            // 3) cusparse direction
            cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

            /* **************************************** */
            // set buffer csr storage
            // 1) set buffer csr rowptr & offset
            cudaMalloc((void **)&temp_csrrowptr, sizeof(int) * (32 + 1));
            if (i == nblockrows - 1)
            { // last iteration
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 32, sizeof(int) * ((nrows + 1) - (i * 32)), cudaMemcpyHostToDevice);
                offsetDeviceArr<int><<<(((nrows + 1) - (i * 32))+1024-1)/1024,1024>>>(temp_csrrowptr, ((nrows + 1) - (i * 32)), temp_rowstart); // offset rowptr
                padDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, ((nrows + 1) - (i * 32)), (32 + 1), temp_nnz);
            }
            else
            { // all other iteration except last
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 32, sizeof(int) * (32 + 1), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, (32 + 1), temp_rowstart); // offset rowptr
            }

            // 2) set buffer csr colind
            cudaMalloc((void **)&temp_csrcolind, sizeof(int) * temp_nnz);
            cudaMemcpy(temp_csrcolind, h_csrColInd + temp_rowstart, sizeof(int) * temp_nnz, cudaMemcpyHostToDevice);

            // 3) set buffer csr val
            cudaMalloc((void **)&temp_csrval, sizeof(float) * temp_nnz);
            setDeviceArr<float><<<(temp_nnz+1024-1)/1024, 1024>>>(temp_csrval, temp_nnz, 1.0);

            // calculate nnzb & allocate buffer bsr storage
            int h_bsrRowPtr_0, h_bsrRowPtr_1;
            cudaMemcpy(&h_bsrRowPtr_0, bsrRowPtr+i, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_bsrRowPtr_1, bsrRowPtr+i+1, sizeof(int), cudaMemcpyDeviceToHost);
            int temp_nblocks = h_bsrRowPtr_1 - h_bsrRowPtr_0;
            cudaMalloc((void **)&temp_bsrrowptr, sizeof(int) * 2);
            setDeviceIndArrElem<int><<<1, 1>>>(temp_bsrrowptr, 0, 0);
            setDeviceIndArrElem<int><<<1, 1>>>(temp_bsrrowptr, 1, temp_nblocks);

            cudaMalloc((void **)&temp_bsrcolind, sizeof(int) * temp_nblocks);
            cudaMalloc((void **)&temp_bsrval, sizeof(float) * temp_nblocks * blocksize * blocksize);

            // csr2bsr (nrows = blocksize, ncols = ncols)
            CHECK_CUSPARSE(cusparseScsr2bsr(handle, dirA, blocksize, ncols, csr_descr,
                                            temp_csrval, temp_csrrowptr, temp_csrcolind,
                                            blocksize, bsr_descr, temp_bsrval, temp_bsrrowptr, temp_bsrcolind));
            total_nblocks += temp_nblocks;

            // pack buffer bsrval to binary
            cudaMalloc((void **)&temp_bsrval_packed, sizeof(unsigned) * temp_nblocks * blocksize);
            ToBit32Col<float><<<dim3(1, temp_nblocks), 32>>>(temp_bsrval,
                                                             temp_bsrval_packed, blocksize, temp_nblocks * blocksize);

            cudaMemcpy(bsrColInd + h_bsrRowPtr_0, temp_bsrcolind, sizeof(int) * temp_nblocks, cudaMemcpyDeviceToDevice);
            cudaMemcpy(bsrVal + h_bsrRowPtr_0 * blocksize, temp_bsrval_packed, sizeof(uchar) * temp_nblocks * blocksize, cudaMemcpyDeviceToDevice);

            // clean buffer
            cudaFree(temp_csrrowptr);
            temp_csrrowptr = NULL;
            cudaFree(temp_csrcolind);
            temp_csrcolind = NULL;
            cudaFree(temp_csrval);
            temp_csrval = NULL;
            cudaFree(temp_bsrrowptr);
            temp_bsrrowptr = NULL;
            cudaFree(temp_bsrcolind);
            temp_bsrcolind = NULL;
            cudaFree(temp_bsrval);
            temp_bsrval = NULL;
            cudaFree(temp_bsrval_packed);
            temp_bsrval_packed = NULL;

            // free descr and handle memory
            CHECK_CUSPARSE(cusparseDestroyMatDescr(csr_descr));
            CHECK_CUSPARSE(cusparseDestroyMatDescr(bsr_descr));
            CHECK_CUSPARSE(cusparseDestroy(handle));
        }
    }
}

/**
* batch the process of csr2bsr, blocksize = 64
* assume csr val are only 0 or 1
*/
void csr2bsr_batch_64(const int *h_csrRowPtr, const int *h_csrColInd,
                      const int nrows, const int ncols, const int nnz,
                      int *bsrRowPtr, int *bsrColInd, ullong *bsrVal,
                      const int blocksize, const int nblockrows, const int nblocks)
{
    // global result
    cudaMemset(bsrColInd, 0, nblocks);
    cudaMemset(bsrVal, 0, nblocks * blocksize);
    int total_nblocks = 0;

    // buffer storage
    int *temp_csrrowptr, *temp_csrcolind;
    float *temp_csrval;
    int *temp_bsrrowptr, *temp_bsrcolind;
    float *temp_bsrval;
    ullong *temp_bsrval_packed;

    for (int i = 0; i < nblockrows; i++)
    { // serial processing, 1 blockrow at a time

        // calculate nnz in this blockrow
        int temp_rowend = ((i + 1) * 64 < nrows ? h_csrRowPtr[(i + 1) * 64] : nnz), temp_rowstart = h_csrRowPtr[i * 64];
        int temp_nnz = temp_rowend - temp_rowstart;

        if (temp_nnz != 0)
        {
            // set up cusparse metadata
            // 1) create cusparsematdescr for csr, bsr
            cusparseMatDescr_t csr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&csr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(csr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(csr_descr, CUSPARSE_INDEX_BASE_ZERO));
            cusparseMatDescr_t bsr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&bsr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(bsr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(bsr_descr, CUSPARSE_INDEX_BASE_ZERO));

            // 2) cusparse handle, stream
            cudaStream_t streamId = 0;
            cusparseHandle_t handle = 0;
            CHECK_CUSPARSE(cusparseCreate(&handle));
            CHECK_CUSPARSE(cusparseSetStream(handle, streamId));

            // 3) cusparse direction
            cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

            /* **************************************** */
            // set buffer csr storage
            // 1) set buffer csr rowptr & offset
            cudaMalloc((void **)&temp_csrrowptr, sizeof(int) * (64 + 1));
            if (i == nblockrows - 1)
            { // last iteration
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 64, sizeof(int) * ((nrows + 1) - (i * 64)), cudaMemcpyHostToDevice);
                offsetDeviceArr<int><<<(((nrows + 1) - (i * 64))+1024-1)/1024,1024>>>(temp_csrrowptr, ((nrows + 1) - (i * 64)), temp_rowstart); // offset rowptr
                padDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, ((nrows + 1) - (i * 64)), (64 + 1), temp_nnz);
            }
            else
            { // all other iteration except last
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 64, sizeof(int) * (64 + 1), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, (64 + 1), temp_rowstart); // offset rowptr
            }

            // 2) set buffer csr colind
            cudaMalloc((void **)&temp_csrcolind, sizeof(int) * temp_nnz);
            cudaMemcpy(temp_csrcolind, h_csrColInd + temp_rowstart, sizeof(int) * temp_nnz, cudaMemcpyHostToDevice);

            // 3) set buffer csr val
            cudaMalloc((void **)&temp_csrval, sizeof(float) * temp_nnz);
            setDeviceArr<float><<<(temp_nnz+1024-1)/1024, 1024>>>(temp_csrval, temp_nnz, 1.0);

            // calculate nnzb & allocate buffer bsr storage
            int h_bsrRowPtr_0, h_bsrRowPtr_1;
            cudaMemcpy(&h_bsrRowPtr_0, bsrRowPtr+i, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_bsrRowPtr_1, bsrRowPtr+i+1, sizeof(int), cudaMemcpyDeviceToHost);
            int temp_nblocks = h_bsrRowPtr_1 - h_bsrRowPtr_0;
            cudaMalloc((void **)&temp_bsrrowptr, sizeof(int) * 2);
            setDeviceIndArrElem<int><<<1, 1>>>(temp_bsrrowptr, 0, 0);
            setDeviceIndArrElem<int><<<1, 1>>>(temp_bsrrowptr, 1, temp_nblocks);

            cudaMalloc((void **)&temp_bsrcolind, sizeof(int) * temp_nblocks);
            cudaMalloc((void **)&temp_bsrval, sizeof(float) * temp_nblocks * blocksize * blocksize);

            // csr2bsr (nrows = blocksize, ncols = ncols)
            CHECK_CUSPARSE(cusparseScsr2bsr(handle, dirA, blocksize, ncols, csr_descr,
                                            temp_csrval, temp_csrrowptr, temp_csrcolind,
                                            blocksize, bsr_descr, temp_bsrval, temp_bsrrowptr, temp_bsrcolind));
            total_nblocks += temp_nblocks;

            // pack buffer bsrval to binary
            cudaMalloc((void **)&temp_bsrval_packed, sizeof(ullong) * temp_nblocks * blocksize);
            ToBit64Col<float><<<dim3(2, temp_nblocks), 32>>>(temp_bsrval,
                                                             temp_bsrval_packed, blocksize, temp_nblocks * blocksize);

            cudaMemcpy(bsrColInd + h_bsrRowPtr_0, temp_bsrcolind, sizeof(int) * temp_nblocks, cudaMemcpyDeviceToDevice);
            cudaMemcpy(bsrVal + h_bsrRowPtr_0 * blocksize, temp_bsrval_packed, sizeof(uchar) * temp_nblocks * blocksize, cudaMemcpyDeviceToDevice);

            // clean buffer
            cudaFree(temp_csrrowptr);
            temp_csrrowptr = NULL;
            cudaFree(temp_csrcolind);
            temp_csrcolind = NULL;
            cudaFree(temp_csrval);
            temp_csrval = NULL;
            cudaFree(temp_bsrrowptr);
            temp_bsrrowptr = NULL;
            cudaFree(temp_bsrcolind);
            temp_bsrcolind = NULL;
            cudaFree(temp_bsrval);
            temp_bsrval = NULL;
            cudaFree(temp_bsrval_packed);
            temp_bsrval_packed = NULL;

            // free descr and handle memory
            CHECK_CUSPARSE(cusparseDestroyMatDescr(csr_descr));
            CHECK_CUSPARSE(cusparseDestroyMatDescr(bsr_descr));
            CHECK_CUSPARSE(cusparseDestroy(handle));
        }
    }
}