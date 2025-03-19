# mad_throughput_benchmark
Slightly modified benchmark of IMAD and FFMA instructions from [njuffa's post](https://forums.developer.nvidia.com/t/blackwell-integer/320578/45) on Blackwell integer throughput.

It was tested on RTX 4080 and RTX 5080. When compiled as
```
nvcc -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_120,code=sm_120 -o throughput.exe throughput.cu
```
the results are as follows:
```
running on device 0 (NVIDIA GeForce RTX 4080)
using 256 threads per block, 524160 blocks, 1.073480 GB used
testing INT32 op throughput with IMAD (one IMAD = two iops)
iops= 6.597338e+12  elapsed=0.24373 sec  throughput=27.06808 Tiops (via IMAD)
testing FP32 op throughput with FMAD (one FMAD = two flops)
flops= 6.597338e+12  elapsed=0.14354 sec  throughput=45.96202 Tflops (via FMAD)

running on device 0 (NVIDIA GeForce RTX 5080)
using 256 threads per block, 524160 blocks, 1.073480 GB used
testing INT32 op throughput with IMAD (one IMAD = two iops)
iops= 6.597338e+12  elapsed=0.21339 sec  throughput=30.91623 Tiops (via IMAD)
testing FP32 op throughput with FMAD (one FMAD = two flops)
flops= 6.597338e+12  elapsed=0.12830 sec  throughput=51.42022 Tflops (via FMAD)
```
[Blackwell whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf) provides the same peak INT32 and FP32 TOPS for RTX 5080 and states that "Blackwell SM provides a doubling of integer math throughput per clock cycle compared to NVIDIA Ada GPUs". It isn't clear at the moment how to achieve this kind of speedup.