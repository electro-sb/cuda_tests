Running the code
```shell
module load cuda
nvcc -o hello hello.cu
```

if already cuda toolkit is installed, then use
```shell
nvcc -o hello hello.cu
```

With profiler
```shell
sudo nsys profile --stats=true ./reductions
nsys stats report.qdrep --report operations
```

## profiling with kernel
```shell
    nsys profile \
    --output=report1 \
    --stats=true \
    -t cuda \
    ./executable
``` 

## Saving file in csv format
```shell
nsys stats report1.qdrep \
  --report=gpu-kernels \
  --format=csv \
  > kernel_times.csv
```
