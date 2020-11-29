# ParallelComputingFinalProject

## How to compile our project
nvcc -arch=sm_50 -lcurand -o main main.cu

## To get debug information, like timers and CUDA hardware specs
main.exe debug