#rm CMakeCache.txt && cmake ..
# Arg 1 is loss (iou for IouLoss, anything else for MSE)
# Arg 2 is modelName (default_model.pt by default)
# Arg 3 is wheter or not to reload --no-reload for no reloading, anything else to reload (if left blank, will not reload)
# Arg 4 is path where preprocessed data tensors should be stored and read from (optional)
cd build
make -j 4

# should be absolute path
INPUT_FILE=$1
DEBUG=$2

if [[ "$DEBUG" == "--debug" ]]; then
    echo "RUNNING IN DEBUG MODE"
    gdb --args ./Open_CV_Project $INPUT_FILE
else
    ./Open_CV_Project $INPUT_FILE
fi
