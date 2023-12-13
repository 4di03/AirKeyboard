
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <ctime>
#include "data_work.h"
#include "train.h"
<<<<<<< HEAD
using namespace std;


void saveSample(float propTrain = 0.1, float propTest = 0.1){
    /**
     * @brief This function  loads data and saves samples o a tensor file in data/data_tensors
     */
    std::string trainData = "/Users/adithyapalle/work/CS5100/Project/data/hanco_all/HanCo/train_keypoint_data.csv";
    Dataset train = prepData(trainData, propTrain);
=======
#include <filesystem>
//#include "model.h"

using namespace std;


void saveSample(float propTrain = 0.1, float propTest = 0.1 , std::string save_folder = "/scratch/palle.a/AirKeyboard/data/data_tensors", bool excludeMerged= false){
    /**
     * @brief This function  loads data and saves samples o a tensor file in data/data_tensors
     */

    std::filesystem::create_directories(save_folder);
    std::string trainData = "/scratch/palle.a/AirKeyboard/data/hanco_all/HanCo/train_keypoint_data.csv";
    Dataset train = prepData(trainData, propTrain, excludeMerged);
>>>>>>> 9faf08865b02c831689cb0fbc1434c782d8b2966

    cout << "train x shape: " << train.x.sizes() << endl;
    cout << "train y shape: " << train.y.sizes() << endl;

    torch::Tensor xTrain = train.x;
    torch::Tensor yTrain = train.y;

<<<<<<< HEAD
    torch::save(xTrain, "/Users/adithyapalle/work/CS5100/Project/data/data_tensors/xTrain.pt");
    torch::save(yTrain, "/Users/adithyapalle/work/CS5100/Project/data/data_tensors/yTrain.pt");

    std::string testData = "/Users/adithyapalle/work/CS5100/Project/data/hanco_all/HanCo/test_keypoint_data.csv";
    Dataset test = prepData(testData, propTest);

    torch::Tensor xTest = test.x;
    torch::Tensor yTest = test.y;
    torch::save(xTest, "/Users/adithyapalle/work/CS5100/Project/data/data_tensors/xTest.pt");
    torch::save(yTest, "/Users/adithyapalle/work/CS5100/Project/data/data_tensors/yTest.pt");
}

std::vector<Dataset> loadSamples(){
=======
    // torch::save(trainMeans, save_folder + "/xMean.pt");
    // torch::save(trainStds, save_folder + "/xStd.pt");

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
>>>>>>> 9faf08865b02c831689cb0fbc1434c782d8b2966
    /**
     * @brief This function loads samples from the tensor files in data/data_tensors
     * @return std::tuple<Dataset>
     */
    torch::Tensor xTrain;
<<<<<<< HEAD
    torch::load(xTrain,"/Users/adithyapalle/work/CS5100/Project/data/data_tensors/xTrain.pt");
    torch::Tensor yTrain;
    torch::load(yTrain,"/Users/adithyapalle/work/CS5100/Project/data/data_tensors/yTrain.pt");


    xTrain = xTrain.permute({0, 3, 1, 2});// initalliy in (N,W,H,C) format, but we need (N,C,W,H)
    yTrain = yTrain.permute({0, 4, 1, 2,3});// initalliy in (N,K,W,H,C) format, but we need (N,C,K,W,H)
    Dataset train = {xTrain, yTrain};

    torch::Tensor xTest;
    torch::load(xTest,"/Users/adithyapalle/work/CS5100/Project/data/data_tensors/xTest.pt");
    torch::Tensor yTest;
    torch::load(yTest,"/Users/adithyapalle/work/CS5100/Project/data/data_tensors/yTest.pt");
    xTest = xTest.permute({0, 3, 1, 2});// initalliy in (N,W,H,C) format, but we need (N,C,W,H)
    yTest = yTest.permute({0, 4, 1, 2,3});// initalliy in (N,K,W,H,C) format, but we need (N,C,K,W,H)
=======
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
>>>>>>> 9faf08865b02c831689cb0fbc1434c782d8b2966
    Dataset test = {xTest, yTest};

    return {train, test};

}
<<<<<<< HEAD

int main() {
    // Open a connection to the webcam (usually 0 for the default webcam)
    //cv::VideoCapture cap(0);
    
    //saveSample(0.01, 0.1);


    auto data = loadSamples();
    Dataset train = data[0];
    Dataset test = data[1];

    cout << "train x shape: " << train.x.sizes() << endl;
    cout << "train y shape: " << train.y.sizes() << endl;
    cout << "test x shape: " << test.x.sizes() << endl;
    cout << "test y shape: " << test.y.sizes() << endl;

    trainModel(train, test);


    // std::cout << "Expected FPS: " << cap.get(cv::CAP_PROP_FPS) << std::endl;
    // int ct = 0;


    // torch::Tensor tensor = torch::rand({2, 3});
    // std::cout << tensor << std::endl;

    // Check if the webcam opened successfully
    // if (!cap.isOpened()) {
    //     std::cerr << "Error: Could not open webcam." << std::endl;
    //     return -1;
    // }

    // // Create a window to display the webcam feed
    // cv::namedWindow("Webcam Feed", cv::WINDOW_NORMAL);

    // // get start time 
    // std::time_t result = std::time(nullptr);


    // // Main loop
    // while (false) {
    //     ct++;
    //     // Capture a frame from the webcam
    //     cv::Mat frame;
    //     cap >> frame;

    //     // Check if the frame is empty
    //     if (frame.empty()) {
    //         std::cerr << "Error: Webcam disconnected or reached end of stream." << std::endl;
    //         break;
    //     }

    //     std::time_t cur_time = std::time(nullptr);
    

    //     float time_diff = cur_time - result;
    //     std::cout << "time diff: " << time_diff << std::endl;
    //     float avg_fps = ct/ time_diff;

        
    //     // Display the frame in the window
    //     cv::imshow("Webcam Feed", frame);

    //     std::cout << "Current FPS: " << avg_fps << std::endl;

    //     // Break the loop if the user presses the 'ESC' key
    //     if (cv::waitKey(1) == 27) {
    //         break;
    //     }
    // }

    // // Release the webcam and close the window
    // cap.release();
    // cv::destroyAllWindows();

    // return 0;
}
=======
// TODO: double check x,y stuff with predictions!!!( especially when averaging heatmap)
int main(int argc, char* argv[]) {
    bool SEED = false;
    
    // ARGUMENTS:
    // arg 1 is loss name , arg 2 is model name, arg 3 is reload option

    // Open a connection to the webcam (usually 0 for the default webcam)

    //saveSample(0.002, 0.03,  "/scratch/palle.a/AirKeyboard/data/data_tensors/samples");
    //saveSample(0.0005, 0.005,  "/scratch/palle.a/AirKeyboard/data/data_tensors/samples");
    if (argc == 1){
        cout << "Provide cmd arguments : arg 1 is loss name , arg 2 is model name, arg 3 is reload option"  << endl;
    }



    std::string modelName = "default_model.pt";
    if (argc > 2){
        modelName = std::string(argv[2]);

        cout << "Model Name: " << modelName <<  endl;

    }

    float propDataUsed = 1;
    bool useCuda = true;


    Loss* loss;
    if (std::string(argv[1]) == "iou"){
        cout << "USING IOU LOSS" << endl;
        loss = new IouLoss();
    }else{
        cout << "USING MSE LOSS" << endl;
        loss =new MSELoss();
    }

    bool reload = false;
    if (argc > 3){
        auto arg3 = std::string(argv[3]);
        cout << "read argv " << arg3 << endl;
         reload = (arg3 != "--no-reload");
    }
 
    std::string dataPath = "/scratch/palle.a/AirKeyboard/data/data_tensors/pure_data";
    if (argc > 4){
        dataPath = std::string(argv[4]);
        cout << "setting data path as " << dataPath << endl;

    }

    float propTrain = 0.02;
    if (argc > 5){
        propTrain= std::stof(std::string(argv[5]));
    }

    float propTest= 0.08;
    if (argc > 6){
        propTest = std::stof(std::string(argv[6]));
    }

    std::string preloadedModelPath = "";

    if (argc > 7){
        preloadedModelPath = std::string(argv[7]);
    }

    if (reload){
        cout << "Reloading data to " << dataPath << endl;
    }else{
    cout << "Not reloading data, pulling from " << dataPath << endl;

    }   

    if (reload){
    bool excludeMerged = false;
    if (excludeMerged){
    cout << "Excluding background swapped images" << endl;
    }
    saveSample(propTrain,propTest, dataPath , excludeMerged);
    }

    if (SEED){
    torch::manual_seed(0);
    torch::cuda::manual_seed(0);
    }

    auto data = loadSamples(dataPath);
    Dataset train = data[0];
    Dataset test = data[1];

    cout << "running Training!" << endl;



    TrainParams tp = TrainParams()
        .setBatchSize(64)
        .setEpochs(500)
        .setNeurons(16)
        .setLevels(6)
        .setCuda(useCuda)
        .setPropDataUsed(propDataUsed)
        .setModelName(modelName)
        .setPropVal(0.1)
        .setStandardize(true)//false);
        .setModelPath(preloadedModelPath)
        .setLoss(loss);

    trainModel(train, test, tp);

    evaluate(test, tp, true);

    // auto device = torch::Device(torch::kCUDA,0);
    // JitModel model("/scratch/palle.a/AirKeyboard/python_sample/weights/model_final", device );
    // auto loss = IouLoss();
    // float loss = evaluateTest(test,  device, model, loss);

    // cout << "TEST LOSS FOR MODEL_FINAL " << loss << endl;


}
>>>>>>> 9faf08865b02c831689cb0fbc1434c782d8b2966
