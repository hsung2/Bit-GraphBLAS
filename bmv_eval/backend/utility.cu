/**
* Filename: utility.cu
*
* Description: utility functions for debugging, profiling on bsrbmv.
*
*/
#include <cusparse_v2.h>
#include <iostream>
#include <sys/time.h>

//======================================================================================
// Error checking for cuda libraries' APIs
//======================================================================================
/**
* Error Checking for cuSparse library
*/
#define CHECK_CUSPARSE(err) __cusparseSafeCall(err, __FILE__, __LINE__)

inline void __cusparseSafeCall(cusparseStatus_t err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
    if (CUSPARSE_STATUS_SUCCESS != err)
    {
        fprintf(stderr, "CUSPARSE API failed at %s:%i : %d\n",
                file, line, err);
        exit(-1);
    }
#endif
    return;
}

//======================================================================================
// Timing functions
//======================================================================================
/**
* The CPU Timer used in GraphBlast
*/
struct CpuTimer
{
#if defined(CLOCK_PROCESS_CPUTIME_ID)

    double start;
    double stop;

    void Start()
    {
        static struct timeval tv;
        static struct timezone tz;
        gettimeofday(&tv, &tz);
        start = tv.tv_sec + 1.e-6 * tv.tv_usec;
    }

    void Stop()
    {
        static struct timeval tv;
        static struct timezone tz;
        gettimeofday(&tv, &tz);
        stop = tv.tv_sec + 1.e-6 * tv.tv_usec;
    }

    double ElapsedMillis()
    {
        return 1000 * (stop - start);
    }

#else

    rusage start;
    rusage stop;

    void Start()
    {
        getrusage(RUSAGE_SELF, &start);
    }

    void Stop()
    {
        getrusage(RUSAGE_SELF, &stop);
    }

    float ElapsedMillis()
    {
        float sec = stop.ru_utime.tv_sec - start.ru_utime.tv_sec;
        float usec = stop.ru_utime.tv_usec - start.ru_utime.tv_usec;

        return (sec * 1000) + (usec / 1000);
    }

#endif
};

/**
* The GPU Timer used in GraphBlast
*/
struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float ElapsedMillis()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

//======================================================================================
// Verification function for result vector
//======================================================================================
/**
* verify bsrbmv result vector with cuSPARSE baseline
*/
template <typename T>
bool checkResult(T *vec1, T *vec2, const int N)
{
    bool flag = true;
    for (int i = 0; i < N; i++)
    {
        T diff = vec1[i] - vec2[i];
        if (fabs(diff) > 1e-6)
        {
            //printf("[%d](%.f,%.f),", i, vec1[i], vec2[i]);
            flag = false;
        }
    }
    return flag;
}

/**
* verify bsrbmv binarized result vector with cuSPARSE baseline
*/
template <typename T>
__global__ void bin2full(T *in, float *out, const int nblockrows, const int tiledim)
{
    for (int i = 0; i < nblockrows; i++)
    {
        T j;
        int t = 0;
        for (j = (1 << (tiledim - 1)); j > 0; j = j / 2)
        {
            if (in[i] & j)
            {
                out[i * tiledim + t] = 1;
                t++;
            }

            else
            {
                out[i * tiledim + t] = 0;
                t++;
            }
        }
    }
}

template <typename T>
bool checkResultBin(T *vec1, T *vec2, const int N)
{
    bool flag = true;
    for (int i = 0; i < N; i++)
    {
        if ((vec1[i] == 0 && vec2[i] == 0) || (vec1[i] != 0 && vec2[i] != 0))
            continue;
        else
        {
            flag = false;
            printf("errorind: %d %f %f\n", i, vec1[i], vec2[i]);
        }
    }
    return flag;
}

template <typename T>
int countNnzinVec(const T *vec, const int N)
{
    int counter = 0;
    for (int i = 0; i < N; i++)
        if (vec[i] != 0)
            counter += 1;
    return counter;
}

template <typename T>
__global__ void printResVec(const T *vec, const int N)
{
    for (int i = 0; i < N; i++)
    {
        printf("%d ", vec[i]);
    }
    printf("\n");
}

__global__ void printResVecFloat(const float *vec, const int N)
{
    for (int i = 0; i < N; i++)
    {
        printf("%f ", vec[i]);
        if (i % 32 == 31)
            printf(" ");
    }
    printf("\n");
}

//======================================================================================
// Print function for host (random) vector
//======================================================================================
void printHostVec(const float *vec, const int N)
{
    for (int i = 0; i < N; i++)
    {
        //printf(vec[i] > 0 ? "1" : "0");
        printf("%f ", vec[i]);
    }
    printf("\n");
}

//======================================================================================
// Print function for binarized vector in device
//======================================================================================
__global__ void printBin8Vec(const uchar *packvec, const int N)
{
    for (int i = 0; i < N; i++)
    {
        uchar j;
        for (j = 1 << 7; j > 0; j = j / 2)
            (packvec[i] & j) ? printf("1") : printf("0");
        printf(" ");
    }
    printf("\n");
}

__global__ void printBin16Vec(const ushort *packvec, const int N)
{
    for (int i = 0; i < N; i++)
    {
        ushort j;
        for (j = 1 << 15; j > 0; j = j / 2)
            (packvec[i] & j) ? printf("1") : printf("0");
        printf(" ");
    }
    printf("\n");
}

__global__ void printBin32Vec(const unsigned *packvec, const int N)
{
    for (int i = 0; i < N; i++)
    {
        unsigned j;
        for (j = 1 << 31; j > 0; j = j / 2)
            (packvec[i] & j) ? printf("1") : printf("0");
        printf(" ");
    }
    printf("\n");
}

__global__ void printBin32Block(const unsigned *packbsrval, const int nblocks, const int blocksize)
{
    for (int i = 0; i < nblocks; i++)
    {
        printf("[%d]\n", i);
        for (int j = 0; j < blocksize; j++)
        {
            unsigned k;
            for (k = 1 << 31; k > 0; k = k / 2)
                (packbsrval[i * blocksize + j] & k) ? printf("1") : printf("0");

            printf("\n");
        }
        printf("\n");
    }
}

__global__ void printBin64Vec(const ullong *packvec, const int N)
{
    for (int i = 0; i < N; i++)
    {
        ullong j;
        for (j = 1ULL << 63; j > 0; j = j / 2)
            (packvec[i] & j) ? printf("1") : printf("0");
    }
    printf("\n");
}

__global__ void printBin64Block(const ullong *packbsrval, const int nblocks, const int blocksize)
{
    for (int i = 0; i < nblocks; i++)
    {
        printf("[%d]\n", i);
        for (int j = 0; j < blocksize; j++)
        {
            ullong k;
            for (k = 1ULL << 63; k > 0; k = k / 2)
                (packbsrval[i * blocksize + j] & k) ? printf("1") : printf("0");

            printf("\n");
        }
        printf("\n");
    }
}

//======================================================================================
// Print function for csr and bsr
//======================================================================================
template <typename Index>
void printHostIndArr(const Index *indarr, const Index N)
{
    for (Index i = 0; i < N; i++)
        printf("[%d]%d ", i, indarr[i]);
    printf("\n");
}

template <typename Index>
__global__ void printDeviceIndArr(const Index *indarr, const Index N)
{
    for (Index i = 0; i < N; i++)
        printf("%d ", indarr[i]);
    printf("\n");
}

__global__ void printGlobalBSRBlock4(const uchar *bsrval, const int blocksize, const int nblocks)
{
    printf("--- global bsr 4 block (bitmap) --- \n");
    for (int b = 0; b < nblocks; b++)
    {
        printf("[%d]\n", b);
        for (int j = 0; j < blocksize; j++)
        {
            for (uchar i = 1 << 7; i > 0; i = i / 2)
            {
                (bsrval[b * blocksize + j] & i) ? printf("1") : printf("0");
            }
            printf("\n");
        }
    }
}

__global__ void printGlobalBSR8(const int *bsrrowptr, const int *bsrcolind, const uchar *bsrval,
                                const int blocksize, const int nblockrows, const int nblocks)
{
    printf("--- global bsr 8 --- \n");
    printf("bsrrowptr: \n");
    for (int i = 0; i < (nblockrows + 1); i++)
    {
        printf("%d ", bsrrowptr[i]);
    }
    printf("\n");
    printf("bsrcolind: \n");
    for (int i = 0; i < nblocks; i++)
    {
        printf("%d ", bsrcolind[i]);
    }
    printf("\n");
    printf("bsrval: \n");
    printf("[0] ");
    for (int j = 0; j < blocksize; j++)
    {
        for (uchar i = 1 << 7; i > 0; i = i / 2)
        {
            (bsrval[0 * blocksize + j] & i) ? printf("1") : printf("0");
        }
        printf(" ");
    }
    printf("\n");
    printf("[%d] ", nblocks - 1);
    for (int j = 0; j < blocksize; j++)
    {
        for (uchar i = 1 << 7; i > 0; i = i / 2)
        {
            (bsrval[(nblocks - 1) * blocksize + j] & i) ? printf("1") : printf("0");
        }
        printf(" ");
    }
    printf("\n");
}

__global__ void printGlobalBSRBlock8(const uchar *bsrval, const int blocksize, const int nblocks)
{
    printf("--- global bsr 8 block (bitmap) --- \n");
    for (int b = 0; b < nblocks; b++)
    {
        printf("[%d]\n", b);
        for (int j = 0; j < blocksize; j++)
        {
            for (uchar i = 1 << 7; i > 0; i = i / 2)
            {
                (bsrval[b * blocksize + j] & i) ? printf("1") : printf("0");
            }
            printf("\n");
        }
    }
}

__global__ void printGlobalBSR16(const int *bsrrowptr, const int *bsrcolind, const ushort *bsrval,
                                 const int blocksize, const int nblockrows, const int nblocks)
{
    printf("--- global bsr 16 --- \n");
    printf("bsrrowptr: \n");
    for (int i = 0; i < (nblockrows + 1); i++)
    {
        printf("%d ", bsrrowptr[i]);
    }
    printf("\n");
    printf("bsrcolind: \n");
    for (int i = 0; i < nblocks; i++)
    {
        printf("%d ", bsrcolind[i]);
    }
    printf("\n");
    printf("bsrval: \n");
    printf("[0] ");
    for (int j = 0; j < blocksize; j++)
    {
        for (ushort i = 1 << 15; i > 0; i = i / 2)
        {
            (bsrval[0 * blocksize + j] & i) ? printf("1") : printf("0");
        }
        printf(" ");
    }
    printf("\n");
    printf("[%d] ", nblocks - 1);
    for (int j = 0; j < blocksize; j++)
    {
        for (ushort i = 1 << 15; i > 0; i = i / 2)
        {
            (bsrval[(nblocks - 1) * blocksize + j] & i) ? printf("1") : printf("0");
        }
        printf(" ");
    }
    printf("\n");
}

__global__ void printGlobalBSRBlock16(const ushort *bsrval, const int blocksize, const int nblocks)
{
    printf("--- global bsr 16 block (bitmap) --- \n");
    for (int b = 0; b < nblocks; b++)
    {
        printf("[%d]\n", b);
        for (int j = 0; j < blocksize; j++)
        {
            for (ushort i = 1 << 15; i > 0; i = i / 2)
            {
                (bsrval[b * blocksize + j] & i) ? printf("1") : printf("0");
            }
            printf("\n");
        }
    }
}

__global__ void printGlobalBSR32(const int *bsrrowptr, const int *bsrcolind, const unsigned *bsrval,
                                 const int blocksize, const int nblockrows, const int nblocks)
{
    printf("--- global bsr 32 --- \n");
    printf("bsrrowptr: \n");
    for (int i = 0; i < (nblockrows + 1); i++)
    {
        printf("%d ", bsrrowptr[i]);
    }
    printf("\n");
    printf("bsrcolind: \n");
    for (int i = 0; i < nblocks; i++)
    {
        printf("%d ", bsrcolind[i]);
    }
    printf("\n");
    printf("bsrval: \n");
    printf("[0] ");
    for (int j = 0; j < blocksize; j++)
    {
        for (unsigned i = 1 << 31; i > 0; i = i / 2)
        {
            (bsrval[0 * blocksize + j] & i) ? printf("1") : printf("0");
        }
        printf(" ");
    }
    printf("\n");
    printf("[%d] ", nblocks - 1);
    for (int j = 0; j < blocksize; j++)
    {
        for (unsigned i = 1 << 31; i > 0; i = i / 2)
        {
            (bsrval[(nblocks - 1) * blocksize + j] & i) ? printf("1") : printf("0");
        }
        printf(" ");
    }
    printf("\n");
}

__global__ void printGlobalBSRBlock32(const unsigned *bsrval, const int blocksize, const int nblocks)
{
    printf("--- global bsr 32 block (bitmap) --- \n");
    for (int b = 0; b < nblocks; b++)
    {
        printf("[%d]\n", b);
        for (int j = 0; j < blocksize; j++)
        {
            for (unsigned i = 1 << 31; i > 0; i = i / 2)
            {
                (bsrval[b * blocksize + j] & i) ? printf("1") : printf("0");
            }
            printf("\n");
        }
    }
}

__global__ void printGlobalBSR64(const int *bsrrowptr, const int *bsrcolind, const ullong *bsrval,
                                 const int blocksize, const int nblockrows, const int nblocks)
{
    printf("--- global bsr 64 --- \n");
    printf("bsrrowptr: \n");
    for (int i = 0; i < (nblockrows + 1); i++)
    {
        printf("%d ", bsrrowptr[i]);
    }
    printf("\n");
    printf("bsrcolind: \n");
    for (int i = 0; i < nblocks; i++)
    {
        printf("%d ", bsrcolind[i]);
    }
    printf("\n");
    printf("bsrval: \n");
    printf("[0] ");
    for (int j = 0; j < blocksize; j++)
    {
        for (ullong i = 1ULL << 63; i > 0; i = i / 2)
        {
            (bsrval[0 * blocksize + j] & i) ? printf("1") : printf("0");
        }
        printf(" ");
    }
    printf("\n");
    printf("[%d] ", nblocks - 1);
    for (int j = 0; j < blocksize; j++)
    {
        for (ullong i = 1ULL << 63; i > 0; i = i / 2)
        {
            (bsrval[(nblocks - 1) * blocksize + j] & i) ? printf("1") : printf("0");
        }
        printf(" ");
    }
    printf("\n");
}

__global__ void printGlobalBSRBlock64(const ullong *bsrval, const int blocksize, const int nblocks)
{
    printf("--- global bsr 32 block (bitmap) --- \n");
    for (int b = 0; b < nblocks; b++)
    {
        printf("[%d]\n", b);
        for (int j = 0; j < blocksize; j++)
        {
            for (ullong i = 1ULL << 63; i > 0; i = i / 2)
            {
                (bsrval[b * blocksize + j] & i) ? printf("1") : printf("0");
            }
            printf("\n");
        }
    }
}

__global__ void printTempBSRVal(const float *bsrval, const int blocksize, const int nblocks)
{
    printf("TempBSRVal: \n");
    for (int i = 0; i < nblocks; i++)
    {
        printf("[%d]\n", i);
        for (int j = 0; j < blocksize; j++)
        {
            for (int k = 0; k < blocksize; k++)
            {
                printf(bsrval[i * blocksize * blocksize + j * blocksize + k] > 0 ? "1" : "0");
            }
            printf("\n");
        }
        printf("\n");
    }
}

//======================================================================================
// Set function for device array
//======================================================================================
/* setting whole arr value */ // use when setting value other than 0
template <typename T>
__global__ void setDeviceIndArr(T *arr, const int N, const T val)
{
    for (T i = 0; i < N; i++)
        arr[i] = val;
}

template <typename T>
__global__ void setDeviceArr(T* arr, const int N, const T val) {
    if (blockIdx.x*1024+threadIdx.x < N) arr[blockIdx.x*1024+threadIdx.x] = val;
}

/* setting single element in arr */
template <typename T>
__global__ void setDeviceIndArrElem(T *arr, const int ind, const T val)
{
    arr[ind] = val;
}

/* setting offset val */ // this is only for b2sr packing program
template <typename Index>
__global__ void offsetDeviceIndArr(Index *indarr, const Index N, const Index temp_rowstart)
{
    for (Index i = 0; i < N; i++)
        indarr[i] -= temp_rowstart;
}

template <typename T>
__global__ void offsetDeviceArr(T* arr, const int N, const T temp_rowstart) {
    if (blockIdx.x*1024+threadIdx.x < N) arr[blockIdx.x*1024+threadIdx.x] -= temp_rowstart;
}

template <typename Index>
__global__ void padDeviceIndArr(Index *indarr, const Index startind, const Index endind, const Index temp_nnz)
{
    for (Index i = startind; i < endind; i++)
        indarr[i] = temp_nnz;
}

template <typename Index, typename T>
__global__ void setDeviceValArr(T *arr, const Index N, const T val)
{
    for (Index i = 0; i < N; i++)
        arr[i] = val;
}

template <typename Index, typename T>
__global__ void setDeviceValArrElem(T *arr, const Index ind, const T val)
{
    arr[ind] = val;
}

//======================================================================================
// Print function for timing report
//======================================================================================
__global__ void printTimeReport(const int *arr, const int N)
{
    float minv = (float)arr[0], maxv = (float)arr[0], sum = 0.0;
    //    printf("======================\n");
    for (int i = 0; i < N; i++)
    {
        //        printf("[%5d] %d\n", i, arr[i]);
        sum += arr[i];
        maxv = max(maxv, (float)arr[i]);
        minv = min(minv, (float)arr[i]);
    }
    printf("======================\n");
    printf("min: %d, max: %d, avg: %.2f\n", (int)minv, (int)maxv, sum / N);
    printf("======================\n");
}

__global__ void printTimenLoadReport(const int *time, const int *load, const int N)
{
    float minv = (float)time[0], maxv = (float)time[0], sum = 0.0;
    float minl = (float)load[0], maxl = (float)load[0], suml = 0.0;
    int maxvi = 0, maxli = 0;
    //    printf("======================\n");
    for (int i = 0; i < N; i++)
    {
        //        printf("[%5d] %d %d\n", i, arr[i], load[i]);
        sum += time[i];
        maxv = max(maxv, (float)time[i]);
        if (maxv == (float)time[i])
            maxvi = i;
        minv = min(minv, (float)time[i]);

        suml += load[i];
        maxl = max(maxl, (float)load[i]);
        if (maxl == (float)load[i])
            maxli = i;
        minl = min(minl, (float)load[i]);
    }

    float mean_t = sum / N, mean_l = suml / N;

    float sd_t = 0.0, sd_l = 0.0;
    for (int i = 0; i < N; i++)
    {
        sd_t += pow((time[i] - mean_t), 2);
        sd_l += pow((load[i] - mean_l), 2);
    }
    sd_t = sqrt(sd_t / N);
    sd_l = sqrt(sd_l / N);

    printf("======================\n");
    printf("min_t: %d, max_t: %d, avg_t: %.2f, sd_t: %.2f\n", (int)minv, (int)maxv, mean_t, sd_t);
    printf("min_l: %d, max_l: %d, avg_l: %.2f, sd_l: %.2f\n", (int)minl, (int)maxl, mean_l, sd_l);
    printf("max_t index: %d, max_l index: %d\n", maxvi, maxli);
    printf("======================\n");
}

//======================================================================================
// Print workload info list
//======================================================================================
__global__ void printWorkloadInfoList(const int *workload_info_list, const int *workload_size_list, const int workloadsize)
{
    int cnt = 0;
    printf("======================\n");
    for (int i = 0; i < workloadsize; i++)
    {
        //        if (i == workloadsize-1 || i == workloadsize -2) {
        printf("%d ", workload_info_list[cnt++]); // row
        printf("%d ", workload_info_list[cnt++]); // row_start
        for (int j = 0; j < workload_size_list[i]; j++)
        {
            printf("%d ", workload_info_list[cnt++]);
        }
        printf("| ");
        //        }
    }
    printf("\n======================\n");
}

__global__ void setWorkloadSizeListAcc(int *workload_size_list_acc, const int *workload_size_list, const int workloadsize)
{
    workload_size_list_acc[0] = 0;
    for (int i = 1; i < workloadsize + 1; i++)
    {
        workload_size_list_acc[i] = workload_size_list_acc[i - 1] + workload_size_list[i - 1] + 2;
    }
}

//======================================================================================
// Print function for bytes
//======================================================================================
// Prints to the provided buffer a nice number of bytes (KB, MB, GB, etc)
void printBytes(unsigned bytes)
{
    const char *suffixes[7];
    suffixes[0] = "B";
    suffixes[1] = "KB";
    suffixes[2] = "MB";
    suffixes[3] = "GB";
    suffixes[4] = "TB";
    suffixes[5] = "PB";
    suffixes[6] = "EB";
    uint s = 0; // which suffix to use
    double count = bytes;
    while (count >= 1024 && s < 7)
    {
        s++;
        count /= 1024;
    }
    if (count - floor(count) == 0.0)
        printf("%d (%s)", (int)count, suffixes[s]);
    else
        printf("%.1f (%s)", count, suffixes[s]);
}

//======================================================================================
// Debug utility for binarized function
//======================================================================================
__global__ void verify32BinResVec(const unsigned *packvec, const float *fullvec, const int N)
{
    printf("incorrect ids: \n");
    for (int i = 0; i < N; i++)
    {
        unsigned j;
        int k = 0;
        for (j = 1 << 31; j > 0; j = j / 2)
        {
            if (((packvec[i] & j) > 0) != (int)(fullvec[i * 32 + k] > 0))
            {
                printf("%d ", i * 32 + k);
                printf("(%d %d) ", (packvec[i] & j), (int)fullvec[i * 32 + k]);
            }
            k++;
        }
        //        printf("\n");

        //        for(int k=0; k<32; k++) printf("%d", (int)fullvec[i*32+k]);
        //        printf("\n");
        //
        //        printf("--------\n");
    }
    printf("\n");
}

//======================================================================================
// Print matrix stats
//======================================================================================
template <typename Index>
__global__ void printMatrixStats_4(const uchar *__restrict__ A, const Index *__restrict__ rowptr,
                                   const Index *__restrict__ colind, const Index nblockrows,
                                   int *tile_per_row,
                                   int *nnz_tile_per_row)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;
    int row = bx * 1024 + tid;

    if (row < nblockrows)
    {
        int row_start, row_end, load = row_end - row_start;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        for (int i = 0; i < 4; i++)
        {
            // tile per row
            tile_per_row[row * 4 + i] = load;
            nnz_tile_per_row[row * 4 + i] = load;

            // nnz tile per row
            for (int j = row_start; j < row_end; j++)
            {
                if (A[j * 4 + i] == 0x00)
                {
                    nnz_tile_per_row[row * 4 + i] -= 1;
                }
            }
        }
    }
}

template <typename Index>
__global__ void printMatrixStats_8(const uchar *__restrict__ A, const Index *__restrict__ rowptr,
                                   const Index *__restrict__ colind, const Index nblockrows,
                                   int *tile_per_row,
                                   int *nnz_tile_per_row)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;
    int row = bx * 1024 + tid;

    if (row < nblockrows)
    {
        int row_start, row_end, load = row_end - row_start;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        for (int i = 0; i < 8; i++)
        {
            // tile per row
            tile_per_row[row * 8 + i] = load;
            nnz_tile_per_row[row * 8 + i] = load;

            // nnz tile per row
            for (int j = row_start; j < row_end; j++)
            {
                if (A[j * 8 + i] == 0x00)
                {
                    nnz_tile_per_row[row * 8 + i] -= 1;
                }
            }
        }
    }
}

template <typename Index>
__global__ void printMatrixStats_16(const ushort *__restrict__ A, const Index *__restrict__ rowptr,
                                    const Index *__restrict__ colind, const Index nblockrows,
                                    int *tile_per_row,
                                    int *nnz_tile_per_row)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;
    int row = bx * 1024 + tid;

    if (row < nblockrows)
    {
        int row_start, row_end, load = row_end - row_start;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        for (int i = 0; i < 16; i++)
        {
            // tile per row
            tile_per_row[row * 16 + i] = load;
            nnz_tile_per_row[row * 16 + i] = load;

            // nnz tile per row
            for (int j = row_start; j < row_end; j++)
            {
                if (A[j * 16 + i] == 0x00)
                {
                    nnz_tile_per_row[row * 16 + i] -= 1;
                }
            }
        }
    }
}

template <typename Index>
__global__ void printMatrixStats_32(const unsigned *__restrict__ A, const Index *__restrict__ rowptr,
                                    const Index *__restrict__ colind, const Index nblockrows,
                                    int *tile_per_row,
                                    int *nnz_tile_per_row)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;
    int row = bx * 1024 + threadIdx.x;

    if (row < nblockrows)
    {
        int row_start, row_end, load = row_end - row_start;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        for (int i = 0; i < 32; i++)
        {
            // tile per row
            tile_per_row[row * 32 + i] = load;
            nnz_tile_per_row[row * 32 + i] = load;

            // nnz tile per row
            for (int j = row_start; j < row_end; j++)
            {
                if (A[j * 32 + i] == 0x00)
                {
                    nnz_tile_per_row[row * 32 + i] -= 1;
                }
            }
        }
    }
}

__global__ void estimatevec(const int *__restrict__ rowptr, const int *__restrict__ colind, const int nrows, const int tiledim, int *res)
{
    int row = blockIdx.x * 1024 + threadIdx.x;

    if (row < nrows)
    {
        int row_start, row_end, load = row_end - row_start;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        int count = 0;
        if (load != 0)
        {
            int last = colind[row_start] / tiledim;
            count = 1;
            for (int i = row_start; i < row_end; i += 1)
            {
                if (last != colind[i] / tiledim)
                {
                    last = colind[i] / tiledim;
                    count += 1;
                }
            }
        }
        atomicAdd(res, count);
    }
}