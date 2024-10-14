source scripts/setup_env.sh  && cd build && make -j 4
# Check if the previous command succeeded
if [ $? -ne 0 ]; then
    echo "Failed to source environment setup and make binaries. Exiting."
    exit 1
fi
cd ..
# see evaluate_input.yaml for sample input file structure
INPUT_FILE=$(realpath $1)

cd build

echo "RUNNING EVLAUATION FOR $INPUT_FILE"

gdb --args ./Evaluate $INPUT_FILE