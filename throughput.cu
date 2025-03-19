#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <iostream>

#define DEVICE_ORDINAL  (0)
#define THREADS_PER_BLK (256)
#define LEN             (65520 * 1024 * 2)
#define STAGES          (128)
#define REPS            (16)
#define ITER            (10)

const int DEPTH = STAGES;

#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
double second (void)
{
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer) {
        hasHighResTimer = QueryPerformanceFrequency (&t);
        oofreq = 1.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }
    if (hasHighResTimer) {
        QueryPerformanceCounter (&t);
        return (double)t.QuadPart * oofreq;
    } else {
        return (double)GetTickCount() * 1.0e-3;
    }
}
#elif defined(__linux__) || defined(__APPLE__)
#include <stddef.h>
#include <sys/time.h>
double second (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}
#else
#error unsupported platform
#endif

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

// Macro to catch CUDA errors in kernel launches
#define CHECK_LAUNCH_ERROR()                                          \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaDeviceSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

template <typename T>
__device__ T mad_mix (T a, T b, T c)
{
    c = a * b + c;
    a = b * c + a;
    b = c * a + b;
    return b;
}

template <typename T>
__global__ void kernel (const T * __restrict__ src, 
                        T * __restrict__ dst, 
                        T a, T b, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    T aa = __sinf(a) * a;
    T bb = __cosf(b) * b;
    T cc = __sinf(b) * b;
    T dd = __cosf(a) * a;
    for (int i = tid; i < len; i += stride) {
        T p = src[i] * aa + bb;
        T q = src[i] * bb + aa;
        T r = src[i] * cc + dd;
        T s = src[i] * dd + cc;
        for (int k = 0; k < REPS; k++) {
#pragma unroll DEPTH
            for (int j = 0; j < DEPTH; j++) {
                p = mad_mix(p, bb, aa);
                q = mad_mix(q, bb, aa);
                r = mad_mix(r, bb, aa);
                s = mad_mix(s, bb, aa);
            }
        }
        dst[i] = p * q * r * s;
    }
}    

int main (void)
{
    double start, stop, elapsed, mintime, nbr_of_mad;
    uint32_t *d_a, *d_b;
    struct cudaDeviceProp props;

    CUDA_SAFE_CALL (cudaGetDeviceProperties (&props, DEVICE_ORDINAL));
    printf ("running on device %d (%s)\n", DEVICE_ORDINAL, props.name);

    /* Allocate memory on device */
    CUDA_SAFE_CALL (cudaMalloc((void**)&d_a, sizeof(d_a[0]) * LEN));
    CUDA_SAFE_CALL (cudaMalloc((void**)&d_b, sizeof(d_b[0]) * LEN));
    
    /* Initialize device memory */
    CUDA_SAFE_CALL (cudaMemset(d_a, 0x00, sizeof(d_a[0]) * LEN)); // zero

    /* Compute execution configuration */
    dim3 dimBlock(THREADS_PER_BLK);
    int threadBlocks = (LEN + (dimBlock.x - 1)) / dimBlock.x;

    dim3 dimGrid(threadBlocks);
    
    printf ("using %d threads per block, %d blocks, %f GB used\n", 
            dimBlock.x, dimGrid.x, 2*1e-9*LEN*sizeof(d_a[0]));

    nbr_of_mad = (DEPTH * REPS * 12.0 + 4.0 + 3.0) * LEN;
    
    printf ("testing INT32 op throughput with IMAD (one IMAD = two iops)\n");
    mintime=1e308;
    for (int k = 0; k < ITER; k++) {
        cudaDeviceSynchronize();
        start = second();
        kernel<uint32_t><<<dimGrid,dimBlock>>>(d_a, d_b, 0x5da07326, 0x5102d832, LEN);
        CHECK_LAUNCH_ERROR();
        stop = second();
        elapsed= stop - start;
        if (elapsed < mintime) { mintime = elapsed; }
    }
    printf ("iops=%13.6e  elapsed=%.5f sec  throughput=%.5f Tiops (via IMAD)\n",
            nbr_of_mad * 2, mintime, nbr_of_mad * 2 * 1e-12 / mintime);

    printf ("testing FP32 op throughput with FMAD (one FMAD = two flops)\n");
    mintime=1e308;
    for (int k = 0; k < ITER; k++) {
        cudaDeviceSynchronize();
        start = second();
        kernel<float><<<dimGrid,dimBlock>>>((float*)d_a, (float*)d_b, 0x5da07326, 0x5102d832, LEN);
        CHECK_LAUNCH_ERROR();
        stop = second();
        elapsed= stop - start;
        if (elapsed < mintime) { mintime = elapsed; }
    }
    printf ("flops=%13.6e  elapsed=%.5f sec  throughput=%.5f Tflops (via FMAD)\n",
            nbr_of_mad * 2, mintime, nbr_of_mad * 2 * 1e-12 / mintime);

    CUDA_SAFE_CALL (cudaFree(d_a));
    CUDA_SAFE_CALL (cudaFree(d_b));

    return EXIT_SUCCESS;
}