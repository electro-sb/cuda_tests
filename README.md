[![Profile](./assets/badges/profile.svg)](https://github.com/electro-sb)
###### Project Information:
<!--- Badges --->
[![Project](./assets/badges/Project.svg)](./README.md)
[![Version](./assets/badges/version.svg)](./pyproject.toml)
###### Metadata:
<!--- Badges --->
[![Git Commit](./assets/badges/git.svg)](./README.md)
[![Last Updated](./assets/badges/updated.svg)](./README.md)
[![Build Status](./assets/badges/build.svg)](./pyproject.toml)
###### Documentation:
<!--- Badges --->
[![Documentation](./assets/badges/docs.svg)](./README.md)
###### License:
<!--- Badges --->
[![License](./assets/badges/license.svg)](./LICENSE.md)

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
