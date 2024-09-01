#rm CMakeCache.txt && cmake ..
# Arg 1 is loss (iou for IouLoss, anything else for MSE)
# Arg 2 is modelName (default_model.pt by default)
# Arg 3 is wheter or not to reload --no-reload for no reloading, anything else to reload (if left blank, will not reload)
# Arg 4 is path where preprocessed data tensors should be stored and read from (optional)
make -j 4


gdb --args ./Open_CV_Project $1 $2 $3 $4 $5 $6 $7

