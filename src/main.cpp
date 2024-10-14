
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <ctime>
#include "data_work.h"
#include "train.h"
#include <filesystem>
#include "constants.h"
#include "model.h"
#include <json.hpp>

using namespace std;


void saveSample(float propTrain = 0.1, float propTest = 0.1 , std::string save_folder = std::string(DATA_PATH) + "/data_tensors", bool excludeMerged= false){
    /**
     * @brief This function  loads data and saves samples o a tensor file in data/data_tensors
     */

    std::filesystem::create_directories(save_folder);
    std::string trainData = std::string(DATA_PATH) + "/hanco_all/HanCo/train_keypoint_data.csv";

    cout << "loading train data from : " << trainData  << " with " << propTrain << " percent of data" <<std::endl;
    Dataset train = prepData(trainData, propTrain, excludeMerged);

    cout << "train x shape: " << train.x.sizes() <<std::endl;
    cout << "train y shape: " << train.y.sizes() <<std::endl;

    torch::Tensor xTrain = train.x;
    torch::Tensor yTrain = train.y;

    // torch::save(trainMeans, save_folder + "/xMean.pt");
    // torch::save(trainStds, save_folder + "/xStd.pt");

    torch::save(xTrain, save_folder + "/xTrain.pt");
    torch::save(yTrain, save_folder + "/yTrain.pt");

    std::string testData = std::string(DATA_PATH) + "/hanco_all/HanCo/test_keypoint_data.csv";
    Dataset test = prepData(testData, propTest,excludeMerged);

    torch::Tensor xTest = test.x;
    torch::Tensor yTest = test.y;
    torch::save(xTest, save_folder + "/xTest.pt");
    torch::save(yTest, save_folder + "/yTest.pt");
}

void torchCudaTest(){

    // Check if CUDA is available
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available. Exiting..." << std::endl;
    }

    // Create a random tensor on the CPU
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << "Original Tensor (CPU):\n" << tensor << std::endl;

    // Move the tensor to the GPU
    tensor = tensor.to(torch::kCUDA);
    std::cout << "Tensor on GPU:\n" << tensor << std::endl;

    // Perform operations on the GPU (e.g., add 1)
    tensor = tensor + 1;
    std::cout << "Modified Tensor on GPU:\n" << tensor << std::endl;

    // Move the tensor back to the CPU for printing
    tensor = tensor.to(torch::kCPU);
    std::cout << "Modified Tensor on CPU:\n" << tensor << std::endl;

}




// TODO: double check x,y stuff with predictions!!!( especially when averaging heatmap)
int main(int argc, char* argv[]) {
    bool SEED = false;
    
    // ARGUMENTS:
    // arg 1 is loss name , arg 2 is model name, arg 3 is reload option

    // Open a connection to the webcam (usually 0 for the default webcam)

    //saveSample(0.002, 0.03,  std::string(DATA_PATH) + "/data_tensors/samples");
    //saveSample(0.0005, 0.005,  std::string(DATA_PATH) + "/data_tensors/samples");
    std::string inputFilePath = std::string(argv[1]);
    cout << "loading json from: " << inputFilePath << std::endl;
    nlohmann::json inputParams = loadJson(inputFilePath);

 
    float propDataUsed = 1;
    bool useCuda = true;
    std::string modelName = getOrDefault(inputParams,"modelName",std::string("default_model.pt"));
    Loss* loss = getLoss(getOrDefault(inputParams,"lossName",std::string("iou")));
    bool reload = getOrDefault(inputParams,"reload",false);
    std::string defaultDataPath= std::string(DATA_PATH) + "/data_tensors/_data";
    std::string dataPath = getOrDefault(inputParams, "dataPath", defaultDataPath);
    float propTrain = getOrDefault(inputParams, "propTrain", 0.000003);
    float propTest = getOrDefault(inputParams, "propTest", 0.0001);



    cout << "Model Name: " << modelName << std::endl;
    if (reload){
        cout << "Reloading data to " << dataPath <<std::endl;
    }else{
        cout << "Not reloading data, pulling from " << dataPath <<std::endl;
    }   

    if (reload){
    bool excludeMerged = false;
    if (excludeMerged){
    cout << "Excluding background swapped images" <<std::endl;
    }
    cout << "RELOADING DATA" << std::endl;
    saveSample(propTrain,propTest, dataPath , excludeMerged);
    }

    if (SEED){
    torch::manual_seed(0);
    torch::cuda::manual_seed(0);
    }

    auto data = loadSamples(dataPath);
    Dataset train = data[0];
    Dataset test = data[1];

    cout << "running Training!" <<std::endl;
    auto sizes =train.x.sizes();

    int channels = sizes[1];
    CuNetBuilder* modelBuilder = new CuNetBuilder();
    modelBuilder->inChannels = channels;
    modelBuilder->outChannels = 21;
    modelBuilder->initNeurons = 64;

    float propTestData = 0.1;

    TrainParams tp = TrainParams(loss, modelBuilder)
        .setBatchSize(64)
        .setEpochs(500)
        .setCuda(useCuda)
        .setPropDataUsed(propDataUsed)
        .setModelName(modelName)
        .setPropVal(propTestData)
        .setStandardize(true);

    trainModel(train, test, tp);

    evaluate(test, tp, modelName, true);

    // auto device = torch::Device(torch::kCUDA,0);
    // JitModel model("/scratch/palle.a/PalmPilot/python_sample/weights/model_final", device );
    // auto loss = IouLoss();
    // float loss = evaluateTest(test,  device, model, loss);

    // cout << "TEST LOSS FOR MODEL_FINAL " << loss <<std::endl;


}
