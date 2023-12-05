
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <ctime>
#include "data_work.h"
#include "train.h"

using namespace std;


void saveSample(float propTrain = 0.1, float propTest = 0.1 , std::string save_folder = "/scratch/palle.a/AirKeyboard/data/data_tensors", bool excludeMerged){
    /**
     * @brief This function  loads data and saves samples o a tensor file in data/data_tensors
     */
    std::string trainData = "/scratch/palle.a/AirKeyboard/data/hanco_all/HanCo/train_keypoint_data.csv";
    Dataset train = prepData(trainData, propTrain, excludeMerged);

    cout << "train x shape: " << train.x.sizes() << endl;
    cout << "train y shape: " << train.y.sizes() << endl;

    torch::Tensor xTrain = train.x;
    torch::Tensor yTrain = train.y;

    torch::save(xTrain, save_folder + "/xTrain.pt");
    torch::save(yTrain, save_folder + "/yTrain.pt");

    std::string testData = "/scratch/palle.a/AirKeyboard/data/hanco_all/HanCo/test_keypoint_data.csv";
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



std::vector<Dataset> loadSamples(std::string save_folder = "/scratch/palle.a/AirKeyboard/data/data_tensors"){
    /**
     * @brief This function loads samples from the tensor files in data/data_tensors
     * @return std::tuple<Dataset>
     */
    torch::Tensor xTrain;
    torch::load(xTrain,save_folder+"/xTrain.pt");
    torch::Tensor yTrain;
    torch::load(yTrain,save_folder+"/yTrain.pt");


    //xTrain = xTrain.permute({0, 3, 1, 2});// initalliy in (N,W,H,C) format, but we need (N,C,W,H)
    //yTrain = yTrain.permute({0, 4, 1, 2,3});// initalliy in (N,K,W,H,C) format, but we need (N,C,K,W,H)
    Dataset train = {xTrain, yTrain};

    torch::Tensor xTest;
    torch::load(xTest,save_folder+"/xTest.pt");
    torch::Tensor yTest;
    torch::load(yTest,save_folder+"/yTest.pt");
   // xTest = xTest.permute({0, 3, 1, 2});// initalliy in (N,W,H,C) format, but we need (N,C,W,H)
  //  yTest = yTest.permute({0, 4, 1, 2,3});// initalliy in (N,K,W,H,C) format, but we need (N,C,K,W,H)
    Dataset test = {xTest, yTest};

    return {train, test};

}
//TODO: add flexibility with loss functions in trainModel
int main(int argc, char* argv[]) {
    // Open a connection to the webcam (usually 0 for the default webcam)

    //saveSample(0.002, 0.03,  "/scratch/palle.a/AirKeyboard/data/data_tensors/samples");
    //saveSample(0.0005, 0.005,  "/scratch/palle.a/AirKeyboard/data/data_tensors/samples");

    cout << "L98 " << std::string(argv[0]) <<std::string(argv[1])  << endl;


    std::string modelName = "default_model.pt";
    if (argc > 2){

        modelName = std::string(argv[2]);

        cout << "Model Name: " << modelName <<  endl;

    }

   

    TrainParams tp = TrainParams().setBatchSize(128).setEpochs(300).setNeurons(64).setLevels(4).setCuda(true).setPropDataUsed(1).setModelName(modelName);

    
    if (std::string(argv[1]) == "iou"){
        cout << "USING IOU LOSS" << endl;
        tp = tp.setLoss(new IouLoss());
    }else{
        cout << "USING MSE LOSS" << endl;
        tp = tp.setLoss(new MSELoss());
    }

    bool reload = false;
    if (argc > 3){
        cout << "Not reloading data!" << endl;

         reload = (std::string(argv[3]) == "--no-reload");
    }


    if (reload){
    saveSample(0.05, 0.05,  "/scratch/palle.a/AirKeyboard/data/data_tensors/full_data");
    }

    torch::manual_seed(0);
    torch::cuda::manual_seed(0);
    auto data = loadSamples("/scratch/palle.a/AirKeyboard/data/data_tensors/full_data");
    Dataset train = data[0];
    Dataset test = data[1];
    bool useCuda = true;

    // // train = train.sample(0.1);
    // // test = test.sample(0.5);

    // cout << "running Training!" << endl;

    trainModel(train, test, tp);
    // //trainModel(train, test, true,1, "sample_MSE.pt");

    evaluate(test, tp, true);





}
