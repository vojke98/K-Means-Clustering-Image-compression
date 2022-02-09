# K-Means-Clustering-Image-compression

*INSTRUCTIONS*

*** SERIAL ***
gcc CPU_Sequential.c -lm -O2 -Wl,-rpath,./ -L./ -l:"libfreeimage.so.3" -o CPU_Sequential
./CPU_Sequential ../images/640x480.png ../out.png 128 50

*** OpenMP ***
module load CUDA
gcc CPU_OpenMP.c -fopenmp -O2 -lm -Wl,-rpath,./ -L./ -l:"libfreeimage.so.3" -o CPU_OpenMP
srun -n1 --cpus-per-task=1 --reservation=fri CPU_OpenMP ../images/640x480.png ../out.png 128 50

*** OpenCL **
module load CUDA
gcc GPU_OpenCL.c -lOpenCL -O2 -lm -Wl,-rpath,./ -L./ -l:"libfreeimage.so.3" -o GPU_OpenCL
srun -n1 -G1 --reservation=fri GPU_OpenCL ../images/640x480.png ../out.png 128 50
