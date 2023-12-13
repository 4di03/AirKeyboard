
module load cmake/3.23.2
module load opencv/3.1.0 #-contrib
module unload cuda/9.2
module load cuda/11.8
module load gcc/11.1.0 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/shared/centos7/cuda/11.8/targets/x86_64-linux/lib/
