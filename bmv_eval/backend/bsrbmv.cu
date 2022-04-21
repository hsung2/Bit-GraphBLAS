#include <stdio.h>
#include <assert.h>

typedef unsigned char uchar;       // 8
typedef unsigned short ushort;     // 16
typedef unsigned long long ullong; // 64

// A faster way to obtain lane id in a warp
#define GET_LANEID              \
    unsigned laneid;            \
    asm("mov.u32 %0, %%laneid;" \
        : "=r"(laneid));

//For higher memory access efficiency
template <typename T>
__device__ __inline__ void store64(const void *addr, T a, T b)
{
    *((float2 *)addr) = make_float2(*(float *)(&a), *(float *)(&b));
}
//For higher memory access efficiency

template <typename T>
__device__ __inline__ void store128(const void *addr, T a, T b, T c, T d)
{
    *((float4 *)addr) = make_float4(*(float *)(&a), *(float *)(&b), *(float *)(&c), *(float *)(&d));
}

//======================================================================================
// bit-packing
//======================================================================================

// col-major packing bit 4
template <typename T>
__global__ void ToBit4Col(const T *__restrict__ A, uchar *B, const int nblocks)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; // ceil(nblocks/64)
    const unsigned bx = blockIdx.x; // 1
    unsigned Bval;
    T f0;

#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        f0 = by * 16 * 64 + i * 16 * 2 + laneid < nblocks * 16 ? A[by * 16 * 64 + i * 16 * 2 + laneid] : 0; // <-- laneid will get consecutive 32 (2-blocks)
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0 > 0 ? 1 : 0));                                    //__brev(__ballot(f0>0));
        if (laneid == i)
            Bval = r0;
    }

    // layout block0 at high-16
    B[by * 4 * 64 + laneid * 4 * 2] = (Bval & 0xF0000000) >> 28;
    B[by * 4 * 64 + laneid * 4 * 2 + 1] = (Bval & 0x0F000000) >> 24;
    B[by * 4 * 64 + laneid * 4 * 2 + 2] = (Bval & 0x00F00000) >> 20;
    B[by * 4 * 64 + laneid * 4 * 2 + 3] = (Bval & 0x000F0000) >> 16;

    // layout block1 at low-16
    B[by * 4 * 64 + laneid * 4 * 2 + 4] = (Bval & 0x0000F000) >> 12;
    B[by * 4 * 64 + laneid * 4 * 2 + 5] = (Bval & 0x00000F00) >> 8;
    B[by * 4 * 64 + laneid * 4 * 2 + 6] = (Bval & 0x000000F0) >> 4;
    B[by * 4 * 64 + laneid * 4 * 2 + 7] = (Bval & 0x0000000F);
}

// row-major packing bit 4
template <typename T>
__global__ void ToBit4Row(const T *__restrict__ A, uchar *B, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < (int)ceil((float)nblockrows / 4))
    {
        unsigned Bval = 0;
        T f0;

#pragma unroll
        for (int i = 0; i < 32; i++)
        {
            if (i % 8 < 4)
                f0 = (T)(0); // high-4 bit remain 0
            else
                f0 = A[bx * 4 * 4 + (i - 4 * ((i / 8) + 1))];

            Bval = (Bval << 1) + (f0 > 0);
        }
        B[bx * 4] = (Bval & 0xFF000000) >> 24;
        B[bx * 4 + 1] = (Bval & 0x00FF0000) >> 16;
        B[bx * 4 + 2] = (Bval & 0x0000FF00) >> 8;
        B[bx * 4 + 3] = Bval & 0x000000FF;
    }
}

// col-major packing bit 8
// process 4 8x8x4 at the same time
template <typename T>
__global__ void ToBit8Col(const T *__restrict__ A, uchar *B, const int nblocks)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; // ceil(nblocks/16)
    const unsigned bx = blockIdx.x; // 1
    unsigned Bval;
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        T f0 = by * 8 * 8 * 4 * 4 + i * 32 + laneid < nblocks * 8 * 8 ? A[by * 8 * 8 * 4 * 4 + i * 32 + laneid] : 0; // <-- laneid will get consecutive 32 (half-block)
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0 > 0 ? 1 : 0));                                             //__brev(__ballot(f0>0));
        if (laneid == i)
            Bval = r0;
    }

    B[by * 8 * 4 * 4 + (laneid / 2) * 8 + laneid % 2 * 4] = (Bval & 0xFF000000) >> 24;
    B[by * 8 * 4 * 4 + (laneid / 2) * 8 + laneid % 2 * 4 + 1] = (Bval & 0x00FF0000) >> 16;
    B[by * 8 * 4 * 4 + (laneid / 2) * 8 + laneid % 2 * 4 + 2] = (Bval & 0x0000FF00) >> 8;
    B[by * 8 * 4 * 4 + (laneid / 2) * 8 + laneid % 2 * 4 + 3] = Bval & 0x000000FF;
}

// row-major packing bit 8
template <typename T>
__global__ void ToBit8Row(const T *__restrict__ A, uchar *B, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < (int)ceil((float)nblockrows / 4))
    {
        unsigned Bval = 0;

#pragma unroll
        for (int i = 0; i < 32; i++)
        {
            T f0 = A[bx * 8 * 4 + i];
            Bval = (Bval << 1) + (f0 > 0);
        }
        B[bx * 4] = (Bval & 0xFF000000) >> 24;
        B[bx * 4 + 1] = (Bval & 0x00FF0000) >> 16;
        B[bx * 4 + 2] = (Bval & 0x0000FF00) >> 8;
        B[bx * 4 + 3] = Bval & 0x000000FF;
    }
}

// col-major packing bit 16
template <typename T>
__global__ void ToBit16Col(const T *__restrict__ A, ushort *B, const int nblocks)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; // ceil(nblocks/4)
    const unsigned bx = blockIdx.x; // 1
    unsigned Bval;
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        T f0 = by * 16 * 16 * 4 + i * 16 * 2 + laneid < nblocks * 16 * 16 ? A[by * 16 * 16 * 4 + i * 16 * 2 + laneid] : 0;
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0 > 0 ? 1 : 0)); //__brev(__ballot(f0>0));

        if (laneid == i)
            Bval = r0;
    }

    B[by * 16 * 4 + laneid * 2] = (Bval & 0xFFFF0000) >> 16;
    B[by * 16 * 4 + laneid * 2 + 1] = (Bval & 0x0000FFFF);
}
// 4 16x16 at the same time

// row-major packing bit 16
template <typename T>
__global__ void ToBit16Row(const T *__restrict__ A, ushort *B, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < (int)ceil((float)nblockrows / 2))
    {
        unsigned Bval = 0;
#pragma unroll
        for (int i = 0; i < 32; i++)
        {
            T f0 = A[bx * 32 + i];
            Bval = (Bval << 1) + (f0 > 0);
        }

        B[bx * 2] = (Bval & 0xFFFF0000) >> 16;
        B[bx * 2 + 1] = (Bval & 0x0000FFFF);
    }
}

// weight should be col-major packing, layout is 32 * (32*numofblocks)
// input should be row-major packing, layout is whatever it is originally

// col-major packing bit 32
template <typename T>
__global__ void ToBit32Col(const T *__restrict__ A, unsigned *B, const int A_height, const int A_width) // blocksize, nblocks * blocksize
{
    GET_LANEID;
    const unsigned by = blockIdx.y; // nblocks
    const unsigned bx = blockIdx.x; // 1
    unsigned Bval;
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        T f0 = A[by * 32 * 32 + i * 32 + laneid];
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0 > 0 ? 1 : 0)); //__brev(__ballot(f0>0));
        if (laneid == i)
            Bval = r0;
    }
    B[by * 32 + laneid] = Bval;
}

// row-major packing bit 32
template <typename T>
__global__ void ToBit32Row(const T *__restrict__ A, unsigned *B, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < nblockrows)
    {
        unsigned Bval = 0;
#pragma unroll
        for (int i = 0; i < 32; i++)
        {
            T f0 = A[bx * 32 + i];
            Bval = (Bval << 1) + (f0 > 0);
        }
        B[bx] = Bval;
    }
}

// col-major packing bit 64
template <typename T>
__global__ void ToBit64Col(const T *__restrict__ A, ullong *B, const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; //nblocks
    const unsigned bx = blockIdx.x; // 2 <- set this
    ullong Bval;
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        T f0 = A[by * 64 * 64 + bx * 64 * 32 + i * 64 + laneid];
        T f1 = A[by * 64 * 64 + bx * 64 * 32 + i * 64 + 32 + laneid];
        unsigned r0 = __ballot_sync(0xFFFFFFFF, f0 > 0 ? 1 : 0);
        unsigned r1 = __ballot_sync(0xFFFFFFFF, f1 > 0 ? 1 : 0);

        //        unsigned r0 = __ballot(f0>0);
        //        unsigned r1 = __ballot(f1>0);

        ullong l0;
        asm volatile("mov.b64 %0, {%1,%2};"
                     : "=l"(l0)
                     : "r"(r0), "r"(r1)); //lo,hi
        if (laneid == i)
            Bval = __brevll(l0);
    }
    B[by * 64 + bx * 32 + laneid] = Bval;
}

// row-major packing bit 64
template <typename T>
__global__ void ToBit64Row(const T *__restrict__ A, ullong *B, const int A_height, const int A_width, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < nblockrows)
    {
        GET_LANEID;

        ullong Bval = 0;
#pragma unroll
        for (int i = 0; i < 64; i++)
        {
            T f0 = A[bx * 64 + i];
            Bval = (Bval << 1) | (f0 > 0);
        }
        B[bx] = Bval;
    }
}

//======================================================================================
// bin-bin-bin
//======================================================================================
__global__ void bmv4_bin_bin_bin(const uchar *__restrict__ A, const uchar *__restrict__ B, uchar *C,
                                 const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx * 8 + (laneid >> 2);

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const uchar *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        uchar *Csub = &(C[bx * 8]);
        register unsigned Cm[1] = {0};

#pragma unroll
        for (int i = 0; i < load; i += 1)
        {
            unsigned r0 = Asub[i * 4 + laneid % 4];
            unsigned r1 = Bsub[(colindsub[i])];
            Cm[0] += __popc(r0 & r1);
        }

        // store
        unsigned r2 = __ballot_sync(0xFFFFFFFF, Cm[0] > 0);
        uchar temp = (uchar)(((__brev(r2) >> (28 - ((laneid >> 2) * 4))) & 0xF) & 0x0F);
        Csub[(laneid >> 2)] |= temp;
    }
}

__global__ void bmv4_bin_bin_bin_1024(const uchar *__restrict__ A, const uchar *__restrict__ B, uchar *C,
                                      const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned warpid = (threadIdx.x >> 5);
    GET_LANEID;
    int row = bx * 8 * 32 + warpid * 8 + (laneid >> 2);

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const uchar *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        uchar *Csub = &(C[bx * 8 * 32 + warpid * 8]);
        register unsigned Cm[1] = {0};

#pragma unroll
        for (int i = 0; i < load; i += 1)
        {
            unsigned r0 = Asub[i * 4 + laneid % 4];
            unsigned r1 = Bsub[(colindsub[i])];
            Cm[0] += __popc(r0 & r1);
        }

        // store
        unsigned r2 = __ballot_sync(0xFFFFFFFF, Cm[0] > 0);
        uchar temp = (uchar)(((__brev(r2) >> (28 - ((laneid >> 2) * 4))) & 0xF) & 0x0F);
        Csub[(laneid >> 2)] |= temp;
    }
}

__global__ void bmv4_bin_bin_bin_new(const uchar *__restrict__ A, const uchar *__restrict__ B, uchar *C,
                                 const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx;

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        __shared__ uchar A_shared[32];
        __shared__ uchar B_shared[8];

        const uchar *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        uchar *Csub = &(C[bx]);
        register unsigned Cm = 0;

#pragma unroll
        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            A_shared[laneid] = (i+laneid/4 < load) ? Asub[(i+(laneid/4))*4+(laneid%4)] : 0;
            B_shared[laneid/4] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]] : 0; // 0-7

            // layout A as 8 uchar into 1 unsigned
            unsigned r0 = (A_shared[0+laneid%4] & 0x0F) << 28 | (A_shared[4+laneid%4] & 0x0F) << 24 | (A_shared[8+laneid%4] & 0x0F) << 20 | (A_shared[12+laneid%4] & 0x0F) << 16 |
                        (A_shared[16+laneid%4] & 0x0F) << 12 | (A_shared[20+laneid%4] & 0x0F) << 8 | (A_shared[24+laneid%4] & 0x0F) << 4 | (A_shared[28+laneid%4] & 0x0F);
            unsigned r1 = (B_shared[0] & 0x0F) << 28 | (B_shared[1] & 0x0F) << 24 | (B_shared[2] & 0x0F) << 20 | (B_shared[3] & 0x0F) << 16 |
                        (B_shared[4] & 0x0F) << 12 | (B_shared[5] & 0x0F) << 8 | (B_shared[6] & 0x0F) << 4 | (B_shared[7] & 0x0F);
            Cm += __popc(r0 & r1);
        }

        // store
        unsigned r2 = __ballot_sync(0xFFFFFFFF, Cm > 0);
        uchar temp = (uchar)(((__brev(r2) >> (28 - ((laneid >> 2) * 4))) & 0xF) & 0x0F);
        Csub[laneid%4] |= temp;
    }
}

__global__ void bmv4_bin_bin_bin_new_1024(const uchar *__restrict__ A, const uchar *__restrict__ B, uchar *C,
                                          const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 32 + warpid;

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        __shared__ uchar A_shared[32*32];
        __shared__ uchar B_shared[32*8];

        const uchar *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        uchar *Csub = &(C[row]);
        register unsigned Cm = 0;

#pragma unroll
        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            A_shared[warpid*32+laneid] = (i+laneid/4 < load) ? Asub[(i+(laneid/4))*4+(laneid%4)] : 0;
            B_shared[warpid*8+laneid/4] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]] : 0; // 0-7

            // layout A as 8 uchar into 1 unsigned
            unsigned r0 = (A_shared[warpid*32+0+laneid%4] & 0x0F) << 28 | (A_shared[warpid*32+4+laneid%4] & 0x0F) << 24 | (A_shared[warpid*32+8+laneid%4] & 0x0F) << 20 | (A_shared[warpid*32+12+laneid%4] & 0x0F) << 16 |
                        (A_shared[warpid*32+16+laneid%4] & 0x0F) << 12 | (A_shared[warpid*32+20+laneid%4] & 0x0F) << 8 | (A_shared[warpid*32+24+laneid%4] & 0x0F) << 4 | (A_shared[warpid*32+28+laneid%4] & 0x0F);
            unsigned r1 = (B_shared[warpid*8+0] & 0x0F) << 28 | (B_shared[warpid*8+1] & 0x0F) << 24 | (B_shared[warpid*8+2] & 0x0F) << 20 | (B_shared[warpid*8+3] & 0x0F) << 16 |
                        (B_shared[warpid*8+4] & 0x0F) << 12 | (B_shared[warpid*8+5] & 0x0F) << 8 | (B_shared[warpid*8+6] & 0x0F) << 4 | (B_shared[warpid*8+7] & 0x0F);
            Cm += __popc(r0 & r1);
        }

        // store
        unsigned r2 = __ballot_sync(0xFFFFFFFF, Cm > 0);
        uchar temp = (uchar)(((__brev(r2) >> (28 - ((laneid >> 2) * 4))) & 0xF) & 0x0F);
        Csub[laneid%4] |= temp;
    }
}

__global__ void bmv8_bin_bin_bin(const uchar *__restrict__ A, const uchar *__restrict__ B, uchar *C,
                                 const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx * 4 + (laneid >> 3);

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 8]);
        const uchar *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        uchar *Csub = &(C[bx * 4]);
        register unsigned Cm[1] = {0};

#pragma unroll
        for (int i = 0; i < load; i += 1)
        {
            unsigned r0 = Asub[i * 8 + laneid % 8];
            unsigned r1 = Bsub[(colindsub[i])];
            Cm[0] += __popc(r0 & r1);
        }

        // store
        unsigned r2 = __ballot_sync(0xFFFFFFFF, Cm[0] > 0);
        uchar temp = (uchar)((((__brev(r2) >> (24 - ((laneid >> 3) * 8))) & 0xFF)));
        Csub[(laneid >> 3)] |= temp;
    }
}

__global__ void bmv8_bin_bin_bin_1024(const uchar *__restrict__ A, const uchar *__restrict__ B, uchar *C,
                                      const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned warpid = (threadIdx.x >> 5);
    GET_LANEID;
    int row = bx * 4 * 32 + warpid * 4 + (laneid >> 3);

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 8]);
        const uchar *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        uchar *Csub = &(C[bx * 4 * 32 + warpid * 4]);
        register unsigned Cm = 0;

#pragma unroll
        for (int i = 0; i < load; i += 1)
        {
            unsigned r0 = Asub[i * 8 + laneid % 8];
            unsigned r1 = Bsub[(colindsub[i])];
            Cm += __popc(r0 & r1);
        }

        // store
        unsigned r2 = __ballot_sync(0xFFFFFFFF, Cm > 0);
        uchar temp = (uchar)((((__brev(r2) >> (24 - ((laneid >> 3) * 8))) & 0xFF)));
        Csub[(laneid >> 3)] |= temp;
    }
}

__global__ void bmv16_bin_bin_bin(const ushort *__restrict__ A, const ushort *__restrict__ B, ushort *C,
                                  const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx * 2 + (laneid >> 4);

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const ushort *Asub = &(A[row_start * 16]);
        const ushort *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        ushort *Csub = &(C[bx * 2]);
        register unsigned Cm[1] = {0};

#pragma unroll
        for (int i = 0; i < load; i += 1)
        {
            unsigned r0 = Asub[i * 16 + laneid % 16];
            unsigned r1 = Bsub[(colindsub[i])];
            Cm[0] += __popc(r0 & r1);
        }

        // store
        unsigned r2 = __ballot_sync(0xFFFFFFFF, Cm[0] > 0);
        ushort temp = (ushort)((((__brev(r2) >> (16 - ((laneid >> 4) * 16))) & 0xFFFF)));
        Csub[(laneid >> 4)] |= temp;
    }
}

__global__ void bmv16_bin_bin_bin_1024(const ushort *__restrict__ A, const ushort *__restrict__ B, ushort *C,
                                       const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned warpid = (threadIdx.x >> 5);
    GET_LANEID;
    int row = bx * 2 * 32 + warpid * 2 + (laneid >> 4);

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const ushort *Asub = &(A[row_start * 16]);
        const ushort *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        ushort *Csub = &(C[bx * 2 * 32 + warpid * 2]);
        register unsigned Cm = 0;

#pragma unroll
        for (int i = 0; i < load; i += 1)
        {
            unsigned r0 = Asub[i * 16 + laneid % 16];
            unsigned r1 = Bsub[(colindsub[i])];
            Cm += __popc(r0 & r1);
        }

        // store
        unsigned r2 = __ballot_sync(0xFFFFFFFF, Cm > 0);
        ushort temp = (ushort)((((__brev(r2) >> (16 - ((laneid >> 4) * 16))) & 0xFFFF)));
        Csub[(laneid >> 4)] |= temp;
    }
}

__global__ void bmv32_bin_bin_bin(const unsigned *__restrict__ A, const unsigned *__restrict__ B, unsigned *C,
                                  const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned warpid = (threadIdx.x >> 5);
    GET_LANEID;
    int row = bx * 32 + warpid;

    if (row < nblockrows)
    {

        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const unsigned *Asub = &(A[row_start * 32]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[bx * 32]);
        register unsigned Cm = 0;

#pragma unroll
        for (int i = 0; i < load; i++)
        {
            unsigned r0 = Asub[i * 32 + laneid];
            unsigned r1 = Bsub[(colindsub[i])];
            Cm += __popc(r0 & r1);
        }

        // store
        unsigned r2 = __ballot_sync(0xFFFFFFFF, Cm > 0);
        Csub[warpid] |= (__brev(r2));
    }
}

//======================================================================================
// bin-bin-full v
//======================================================================================
__global__ void bmv4_bin_bin_full(const uchar *__restrict__ A, const uchar *__restrict__ B, float *C,
                                  const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx * 8 + (laneid >> 2);

    if (row < nblockrows)
    {
        int row_start = 0, row_end = 0, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[(row_start * 4)]);
        const uchar *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[bx * 32]);
        register unsigned Cm[1] = {0};

        // compute 4 blocks on 4 consecutive blockrow at a time
        int i;
        for (i = 0; i < (int)ceil((float)load / 4) * 4 - 4; i += 4)
        {
            uchar a0 = Asub[i * 4 + (laneid % 4)];
            uchar a1 = Asub[i * 4 + 4 + (laneid % 4)];
            uchar a2 = Asub[i * 4 + 8 + (laneid % 4)];
            uchar a3 = Asub[i * 4 + 12 + (laneid % 4)];
            unsigned r0 = a0 << 24 | a1 << 16 | a2 << 8 | a3;

            uchar b0 = Bsub[colindsub[i]];
            uchar b1 = Bsub[colindsub[i + 1]];
            uchar b2 = Bsub[colindsub[i + 2]];
            uchar b3 = Bsub[colindsub[i + 3]];
            unsigned r1 = b0 << 24 | b1 << 16 | b2 << 8 | b3;

            Cm[0] += __popc(r0 & r1);
        }

        {
            uchar a0 = i * 4 + (laneid % 4) < load * 4 ? Asub[i * 4 + (laneid % 4)] : 0;
            uchar a1 = i * 4 + 4 + (laneid % 4) < load * 4 ? Asub[i * 4 + 4 + (laneid % 4)] : 0;
            uchar a2 = i * 4 + 8 + (laneid % 4) < load * 4 ? Asub[i * 4 + 8 + (laneid % 4)] : 0;
            uchar a3 = i * 4 + 12 + (laneid % 4) < load * 4 ? Asub[i * 4 + 12 + (laneid % 4)] : 0;
            unsigned r0 = a0 << 24 | a1 << 16 | a2 << 8 | a3;

            uchar b0 = i * 4 + (laneid % 4) < load * 4 ? Bsub[colindsub[i]] : 0;
            uchar b1 = i * 4 + 4 + (laneid % 4) < load * 4 ? Bsub[colindsub[i + 1]] : 0;
            uchar b2 = i * 4 + 8 + (laneid % 4) < load * 4 ? Bsub[colindsub[i + 2]] : 0;
            uchar b3 = i * 4 + 12 + (laneid % 4) < load * 4 ? Bsub[colindsub[i + 3]] : 0;
            unsigned r1 = b0 << 24 | b1 << 16 | b2 << 8 | b3;

            Cm[0] += __popc(r0 & r1);
        }

        // store
        Csub[laneid] += (float)(Cm[0]);
    }
}

__global__ void bmv8_bin_bin_full(const uchar *__restrict__ A, const uchar *__restrict__ B, float *C,
                                  const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx * 4 + (laneid >> 3);

    if (row < nblockrows)
    {
        int row_start = 0, row_end = 0, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 8]);
        const uchar *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[bx * 32]);
        register unsigned Cm[1] = {0};

        // compute 4 blocks on 4 consecutive blockrow at a time
        int i;
        for (i = 0; i < (int)ceil((float)load / 4) * 4 - 4; i += 4)
        {
            uchar a0 = Asub[i * 8 + (laneid % 8)];
            uchar a1 = Asub[i * 8 + 8 + (laneid % 8)];
            uchar a2 = Asub[i * 8 + 16 + (laneid % 8)];
            uchar a3 = Asub[i * 8 + 24 + (laneid % 8)];
            unsigned r0 = a0 << 24 | a1 << 16 | a2 << 8 | a3;

            uchar b0 = Bsub[colindsub[i]];
            uchar b1 = Bsub[colindsub[i + 1]];
            uchar b2 = Bsub[colindsub[i + 2]];
            uchar b3 = Bsub[colindsub[i + 3]];
            unsigned r1 = b0 << 24 | b1 << 16 | b2 << 8 | b3;

            Cm[0] += __popc(r0 & r1);
        }

        {
            uchar a0 = i * 8 + (laneid % 8) < load * 8 ? Asub[i * 8 + (laneid % 8)] : 0;
            uchar a1 = i * 8 + 8 + (laneid % 8) < load * 8 ? Asub[i * 8 + 8 + (laneid % 8)] : 0;
            uchar a2 = i * 8 + 16 + (laneid % 8) < load * 8 ? Asub[i * 8 + 16 + (laneid % 8)] : 0;
            uchar a3 = i * 8 + 24 + (laneid % 8) < load * 8 ? Asub[i * 8 + 24 + (laneid % 8)] : 0;
            unsigned r0 = a0 << 24 | a1 << 16 | a2 << 8 | a3;

            uchar b0 = i * 8 + (laneid % 8) < load * 8 ? Bsub[colindsub[i]] : 0;
            uchar b1 = i * 8 + 8 + (laneid % 8) < load * 8 ? Bsub[colindsub[i + 1]] : 0;
            uchar b2 = i * 8 + 16 + (laneid % 8) < load * 8 ? Bsub[colindsub[i + 2]] : 0;
            uchar b3 = i * 8 + 24 + (laneid % 8) < load * 8 ? Bsub[colindsub[i + 3]] : 0;
            unsigned r1 = b0 << 24 | b1 << 16 | b2 << 8 | b3;

            Cm[0] += __popc(r0 & r1);
        }

        // store
        Csub[laneid] += (float)(Cm[0]);
    }
}

__global__ void bmv16_bin_bin_full(const ushort *__restrict__ A, const ushort *__restrict__ B, float *C,
                                   const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx * 2 + (laneid >> 4);

    if (row < nblockrows)
    {
        int row_start = 0, row_end = 0, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const ushort *Asub = &(A[row_start * 16]);
        const ushort *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[bx * 32]);
        register unsigned Cm[1] = {0};

        // compute 2 blocks on 2 consecutive blockrow at a time
        int i;
        for (i = 0; i < (int)ceil((float)load / 2) * 2 - 2; i += 2)
        {
            ushort a0 = Asub[i * 16 + (laneid % 16)];
            ushort a1 = Asub[i * 16 + 16 + (laneid % 16)];
            unsigned r0 = a0 << 16 | a1;

            ushort b0 = Bsub[colindsub[i]];
            ushort b1 = Bsub[colindsub[i + 1]];
            unsigned r1 = b0 << 16 | b1;

            Cm[0] += __popc(r0 & r1);
        }
        {
            ushort a0 = i * 16 + (laneid % 16) < load * 16 ? Asub[i * 16 + (laneid % 16)] : 0;
            ushort a1 = i * 16 + 16 + (laneid % 16) < load * 16 ? Asub[i * 16 + 16 + (laneid % 16)] : 0;
            unsigned r0 = a0 << 16 | a1;

            ushort b0 = i * 16 + (laneid % 16) < load * 16 ? Bsub[colindsub[i]] : 0;
            ushort b1 = i * 16 + 16 + (laneid % 16) < load * 16 ? Bsub[colindsub[i + 1]] : 0;
            unsigned r1 = b0 << 16 | b1;

            Cm[0] += __popc(r0 & r1);
        }

        // store
        Csub[laneid] += (float)(Cm[0]);
    }
}

__global__ void bmv32_bin_bin_full(const unsigned *__restrict__ A, const unsigned *__restrict__ B, float *C,
                                   const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    int row = bx;

    if (row < nblockrows)
    {
        // retrive index
        GET_LANEID;
        int row_start = rowptr[row];
        int row_end = rowptr[row + 1];

        // do some prefetching
        const unsigned *Asub = &(A[row_start * 32]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[bx * 32]);

        // accumulator
        register unsigned Cm[1] = {0};

        for (int i = row_start; i < row_end; i++)
        {
            unsigned r0 = Asub[(i - row_start) * 32 + laneid];
            unsigned r1 = Bsub[(colindsub[i - row_start])];

            Cm[0] += __popc(r0 & r1);
        }

        Csub[laneid] += (float)(Cm[0]);
    }
}

//======================================================================================
// bin-full-full v
//======================================================================================
__global__ void bmv4_bin_full_full(const uchar *__restrict__ A, const float *__restrict__ B, float *C,
                                   const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned warpid = (threadIdx.x >> 5);
    GET_LANEID;
    int row = bx * 32 + warpid;
    __shared__ float shared_B[128 * 32]; // 32 * 4 T = 128

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[bx * 4 * 32]);
        float sum = 0;

        int i;
        for (i = 0; i < ((load + 32 - 1) / 32) * 32 - 32; i += 32)
        {
            ushort a0 = Asub[i * 4 + (laneid / 4) * 16 + (laneid % 4)];
            ushort a1 = Asub[i * 4 + (laneid / 4) * 16 + 4 + (laneid % 4)];
            ushort a2 = Asub[i * 4 + (laneid / 4) * 16 + 8 + (laneid % 4)];
            ushort a3 = Asub[i * 4 + (laneid / 4) * 16 + 12 + (laneid % 4)];
            ushort r0 = a0 << 12 | a1 << 8 | a2 << 4 | a3;

            store128((void *)&shared_B[warpid * 128 + laneid * 4],
                     (Bsub[(colindsub[i + laneid]) * 4]),
                     (Bsub[(colindsub[i + laneid]) * 4 + 1]),
                     (Bsub[(colindsub[i + laneid]) * 4 + 2]),
                     (Bsub[(colindsub[i + laneid]) * 4 + 3]));
            __syncthreads();

#pragma unroll
            for (int j = 0; j < 16; j++)
            {
                if ((r0 >> (15 - j)) & 0x1)
                    sum += (shared_B[warpid * 128 + (laneid / 4) * 16 + j]);
            }
        }

        {
            ushort a0 = i * 4 + (laneid / 4) * 16 + (laneid % 4) < load * 4 ? Asub[i * 4 + (laneid / 4) * 16 + (laneid % 4)] : 0;
            ushort a1 = i * 4 + (laneid / 4) * 16 + 4 + (laneid % 4) < load * 4 ? Asub[i * 4 + (laneid / 4) * 16 + 4 + (laneid % 4)] : 0;
            ushort a2 = i * 4 + (laneid / 4) * 16 + 8 + (laneid % 4) < load * 4 ? Asub[i * 4 + (laneid / 4) * 16 + 8 + (laneid % 4)] : 0;
            ushort a3 = i * 4 + (laneid / 4) * 16 + 12 + (laneid % 4) < load * 4 ? Asub[i * 4 + (laneid / 4) * 16 + 12 + (laneid % 4)] : 0;
            ushort r0 = a0 << 12 | a1 << 8 | a2 << 4 | a3;

            store128((void *)&shared_B[warpid * 128 + laneid * 4],
                     (i * 4 + (laneid / 4) * 16 + (laneid % 4) < load * 4 ? Bsub[(colindsub[i + laneid]) * 4] : 0),
                     (i * 4 + (laneid / 4) * 16 + (laneid % 4) < load * 4 ? Bsub[(colindsub[i + laneid]) * 4 + 1] : 0),
                     (i * 4 + (laneid / 4) * 16 + (laneid % 4) < load * 4 ? Bsub[(colindsub[i + laneid]) * 4 + 2] : 0),
                     (i * 4 + (laneid / 4) * 16 + (laneid % 4) < load * 4 ? Bsub[(colindsub[i + laneid]) * 4 + 3] : 0));
            __syncthreads();

#pragma unroll
            for (int j = 0; j < 16; j++)
            {
                if ((r0 >> (15 - j)) & 0x1)
                    sum += (shared_B[warpid * 128 + (laneid / 4) * 16 + j]);
            }
        }

        // store
        atomicAdd(Csub + warpid * 4 + laneid % 4, sum);
    }
}

__global__ void bmv8_bin_full_full(const uchar *__restrict__ A, const float *__restrict__ B, float *C,
                                   const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned warpid = (threadIdx.x >> 5);
    GET_LANEID;
    int row = bx * 32 + warpid;
    __shared__ float shared_B[128 * 32];

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 8]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[bx * 8 * 32]);
        float sum = 0;

        int i;
        for (i = 0; i < ((load + 16 - 1) / 16) * 16 - 16; i += 16)
        {
            unsigned a0 = Asub[i * 8 + (laneid / 8) * 32 + (laneid % 8)];
            unsigned a1 = Asub[i * 8 + (laneid / 8) * 32 + 8 + (laneid % 8)];
            unsigned a2 = Asub[i * 8 + (laneid / 8) * 32 + 16 + (laneid % 8)];
            unsigned a3 = Asub[i * 8 + (laneid / 8) * 32 + 24 + (laneid % 8)];
            unsigned r0 = a0 << 24 | a1 << 16 | a2 << 8 | a3;

            int jind = (colindsub[i + laneid / 2]) * 8 + (laneid % 2) * 4;
            store128((void *)&shared_B[warpid * 128 + (laneid / 2) * 8 + (laneid % 2) * 4],
                     (Bsub[jind]),
                     (Bsub[jind + 1]),
                     (Bsub[jind + 2]),
                     (Bsub[jind + 3]));
            __syncthreads();

#pragma unroll
            for (int j = 0; j < 32; j++)
            {
                if ((r0 >> (31 - j)) & 0x1)
                    sum += (shared_B[warpid * 128 + (laneid / 8) * 32 + j]);
            }
        }

        // less than 16
        {
            unsigned a0 = i * 8 + (laneid / 8) * 32 + (laneid % 8) < load * 8 ? Asub[i * 8 + (laneid / 8) * 32 + (laneid % 8)] : 0;
            unsigned a1 = i * 8 + (laneid / 8) * 32 + 8 + (laneid % 8) < load * 8 ? Asub[i * 8 + (laneid / 8) * 32 + 8 + (laneid % 8)] : 0;
            unsigned a2 = i * 8 + (laneid / 8) * 32 + 16 + (laneid % 8) < load * 8 ? Asub[i * 8 + (laneid / 8) * 32 + 16 + (laneid % 8)] : 0;
            unsigned a3 = i * 8 + (laneid / 8) * 32 + 24 + (laneid % 8) < load * 8 ? Asub[i * 8 + (laneid / 8) * 32 + 24 + (laneid % 8)] : 0;
            unsigned r0 = a0 << 24 | a1 << 16 | a2 << 8 | a3;

            int jind = (colindsub[i + laneid / 2]) * 8 + (laneid % 2) * 4;
            store128((void *)&shared_B[warpid * 128 + (laneid / 2) * 8 + (laneid % 2) * 4],
                     (i + laneid / 2 < load ? Bsub[jind] : 0),
                     (i + laneid / 2 < load ? Bsub[jind + 1] : 0),
                     (i + laneid / 2 < load ? Bsub[jind + 2] : 0),
                     (i + laneid / 2 < load ? Bsub[jind + 3] : 0));
            __syncthreads();

#pragma unroll
            for (int j = 0; j < 32; j++)
            {
                if ((r0 >> (31 - j)) & 0x1)
                    sum += (shared_B[warpid * 128 + (laneid / 8) * 32 + j]);
            }
        }

        // store
        atomicAdd(Csub + warpid * 8 + laneid % 8, sum);
    }
}

__global__ void bmv16_bin_full_full(const ushort *__restrict__ A, const float *__restrict__ B, float *C,
                                    const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned warpid = (threadIdx.x >> 5);
    GET_LANEID;
    int row = bx * 32 + warpid;
    __shared__ float shared_B[64 * 32];

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const ushort *Asub = &(A[row_start * 16]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[bx * 16 * 32]);
        float sum = 0;

        int i;
        for (i = 0; i < ((load + 4 - 1) / 4) * 4 - 4; i += 4)
        {
            unsigned a0 = Asub[i * 16 + (laneid / 16) * 32 + (laneid % 16)];
            unsigned a1 = Asub[i * 16 + (laneid / 16) * 32 + 16 + (laneid % 16)];
            unsigned r0 = a0 << 16 | a1;

            int jind = (colindsub[i + laneid / 8]) * 16 + laneid % 8;
            shared_B[warpid * 64 + (laneid / 8) * 16 + laneid % 8] = Bsub[jind];
            shared_B[warpid * 64 + (laneid / 8) * 16 + 8 + laneid % 8] = Bsub[jind + 8];
            __syncthreads();

#pragma unroll
            for (int j = 0; j < 32; j++)
            {
                if ((r0 >> (31 - j)) & 0x1)
                    sum += shared_B[warpid * 64 + (laneid / 16) * 32 + j];
            }
        }

        // less than 4
        {
            unsigned a0 = i * 16 + (laneid / 16) * 32 + (laneid % 16) < load * 16 ? Asub[i * 16 + (laneid / 16) * 32 + (laneid % 16)] : 0;
            unsigned a1 = i * 16 + (laneid / 16) * 32 + 16 + (laneid % 16) < load * 16 ? Asub[i * 16 + (laneid / 16) * 32 + 16 + (laneid % 16)] : 0;
            unsigned r0 = a0 << 16 | a1;

            shared_B[warpid * 64 + (laneid / 8) * 16 + laneid % 8] = i + laneid / 8 < load ? Bsub[(colindsub[i + laneid / 8]) * 16 + laneid % 8] : 0;
            shared_B[warpid * 64 + (laneid / 8) * 16 + 8 + laneid % 8] = i + laneid / 8 < load ? Bsub[(colindsub[i + laneid / 8]) * 16 + 8 + laneid % 8] : 0;
            __syncthreads();

#pragma unroll
            for (int j = 0; j < 32; j++)
            {
                if ((r0 >> (31 - j)) & 0x1)
                    sum += shared_B[warpid * 64 + (laneid / 16) * 32 + j];
            }
        }

        // store
        atomicAdd(Csub + warpid * 16 + laneid % 16, sum);
    }
}

__global__ void bmv32_bin_full_full(const unsigned *__restrict__ A, const float *__restrict__ B, float *C,
                                    const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned warpid = (threadIdx.x >> 5);
    GET_LANEID;
    int row = bx * 32 + warpid;
    __shared__ float shared_B[32 * 32];

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const unsigned *Asub = &(A[row_start * 32]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[bx * 1024]);
        float sum = 0;

#pragma unroll
        for (int i = 0; i < load; i++)
        {
            unsigned r0 = Asub[i * 32 + laneid];

            shared_B[warpid * 32 + laneid] = Bsub[colindsub[i] * 32 + laneid];
            __syncthreads();

#pragma unroll
            for (int j = 0; j < 32; j++)
            {
                if ((r0 >> (31 - j)) & 0x1)
                    sum += (shared_B[warpid * 32 + j]);
            }
        }

        // store
        Csub[warpid * 32 + laneid] += sum;
    }
}

//======================================================================================
// bin-bin-bin-masked
//======================================================================================
__global__ void bmv4_bin_bin_bin_masked(const uchar *__restrict__ A, const uchar *__restrict__ B, uchar *C,
                                        const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                        const uchar *__restrict__ mask)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx * 8 + (laneid >> 2);

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const uchar *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        uchar *Csub = &(C[bx * 8]);
        register unsigned Cm[1] = {0};

#pragma unroll
        for (int i = 0; i < load; i += 1)
        {
            unsigned r0 = Asub[i * 4 + laneid % 4];
            unsigned r1 = Bsub[(colindsub[i])];
            Cm[0] += __popc(r0 & r1);
        }

        // store
        unsigned r2 = __ballot_sync(0xFFFFFFFF, Cm[0] > 0 ? 1 : 0);
        uchar temp = (uchar)((((__brev(r2) >> (28 - ((laneid >> 2) * 4))) & 0xF) & (~mask[row])) & 0x0F);
        Csub[(laneid >> 2)] = temp;
    }
}

__global__ void bmv8_bin_bin_bin_masked(const uchar *__restrict__ A, const uchar *__restrict__ B, uchar *C,
                                        const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                        const uchar *__restrict__ mask)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx * 4 + (laneid >> 3);

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 8]);
        const uchar *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        uchar *Csub = &(C[bx * 4]);
        register unsigned Cm[1] = {0};

#pragma unroll
        for (int i = 0; i < load; i += 1)
        {
            unsigned r0 = Asub[i * 8 + laneid % 8];
            unsigned r1 = Bsub[(colindsub[i])];
            Cm[0] += __popc(r0 & r1);
        }

        // store
        unsigned r2 = __ballot_sync(0xFFFFFFFF, Cm[0] > 0 ? 1 : 0);
        uchar temp = (uchar)((((__brev(r2) >> (32 - ((laneid >> 3) * 8))) & 0xF) & (~mask[row])));
        Csub[(laneid >> 3)] = temp;
    }
}

__global__ void bmv16_bin_bin_bin_masked(const ushort *__restrict__ A, const ushort *__restrict__ B, ushort *C,
                                         const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                         const ushort *__restrict__ mask)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx * 2 + (laneid >> 4);

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const ushort *Asub = &(A[row_start * 16]);
        const ushort *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        ushort *Csub = &(C[bx * 2]);
        register unsigned Cm[1] = {0};

#pragma unroll
        for (int i = 0; i < load; i += 1)
        {
            unsigned r0 = Asub[i * 16 + laneid % 16];
            unsigned r1 = Bsub[(colindsub[i])];
            Cm[0] += __popc(r0 & r1);
        }

        // store
        unsigned r2 = __ballot_sync(0xFFFFFFFF, Cm[0] > 0 ? 1 : 0);
        ushort temp = (ushort)((((__brev(r2) >> (32 - ((laneid >> 4) * 16))) & 0xF) & (~mask[row])));
        Csub[(laneid >> 4)] = temp;
    }
}

__global__ void bmv32_bin_bin_bin_masked(const unsigned *__restrict__ A, const unsigned *__restrict__ B, unsigned *C,
                                         const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                         const unsigned *__restrict__ mask)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned warpid = (threadIdx.x >> 5);
    GET_LANEID;
    int row = bx * 32 + warpid;

    if (row < nblockrows)
    {

        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const unsigned *Asub = &(A[row_start * 32]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[bx * 32]);
        register unsigned Cm[1] = {0};

#pragma unroll
        for (int i = 0; i < load; i++)
        {
            unsigned r0 = Asub[i * 32 + laneid];
            unsigned r1 = Bsub[(colindsub[i])];
            Cm[0] += __popc(r0 & r1);
        }

        // store
        unsigned r2 = __ballot_sync(0xFFFFFFFF, (unsigned)Cm[0] > 0 ? 1 : 0);
        Csub[warpid] = (__brev(r2) & (~mask[bx * 32 + warpid]));
    }
}

//======================================================================================
// bin-bin-full-masked
//======================================================================================
// not actually used, plan to mask 1 element at store

//======================================================================================
// bin-full-full-masked
//======================================================================================
// not actually used, plan to mask 1 element at store