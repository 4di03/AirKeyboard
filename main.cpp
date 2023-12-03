
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <ctime>
#include "data_work.h"
#include "train.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>  // Include xrandom header for randn function
using namespace std;


void saveSample(float propTrain = 0.1, float propTest = 0.1 , std::string save_folder = "/scratch/palle.a/AirKeyboard/data/data_tensors"){
    /**
     * @brief This function  loads data and saves samples o a tensor file in data/data_tensors
     */
    std::string trainData = "/scratch/palle.a/AirKeyboard/data/hanco_all/HanCo/train_keypoint_data.csv";
    Dataset train = prepData(trainData, propTrain);

    cout << "train x shape: " << train.x.sizes() << endl;
    cout << "train y shape: " << train.y.sizes() << endl;

    torch::Tensor xTrain = train.x;
    torch::Tensor yTrain = train.y;

    torch::save(xTrain, save_folder + "/xTrain.pt");
    torch::save(yTrain, save_folder + "/yTrain.pt");

    std::string testData = "/scratch/palle.a/AirKeyboard/data/hanco_all/HanCo/test_keypoint_data.csv";
    Dataset test = prepData(testData, propTest);

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

void xTensorTest(){
        // Create a random 2D array with dimensions 3x4
    xt::xarray<double> array = xt::random::randn<double>({3, 4});

    // Print the original array
    std::cout << "Original Array:\n" << array << std::endl;

    // Perform operations on the array (e.g., element-wise multiplication)
    array = array * 2.0;

    // Print the modified array
    std::cout << "Modified Array (element-wise multiplication by 2.0):\n" << array << std::endl;

    // Accessing elements
    double element = array(1, 2);
    std::cout << "Element at (1, 2): " << element << std::endl;

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


    xTrain = xTrain.permute({0, 3, 1, 2});// initalliy in (N,W,H,C) format, but we need (N,C,W,H)
    yTrain = yTrain.permute({0, 4, 1, 2,3});// initalliy in (N,K,W,H,C) format, but we need (N,C,K,W,H)
    Dataset train = {xTrain, yTrain};

    torch::Tensor xTest;
    torch::load(xTest,save_folder+"/xTest.pt");
    torch::Tensor yTest;
    torch::load(yTest,save_folder+"/yTest.pt");
    xTest = xTest.permute({0, 3, 1, 2});// initalliy in (N,W,H,C) format, but we need (N,C,W,H)
    yTest = yTest.permute({0, 4, 1, 2,3});// initalliy in (N,K,W,H,C) format, but we need (N,C,K,W,H)
    Dataset test = {xTest, yTest};

    return {train, test};

}
//TODO: add flexibility with loss functions in trainModel
int main() {
    // Open a connection to the webcam (usually 0 for the default webcam)
    //cv::VideoCapture cap(0);
    
    //saveSample(0.002, 0.03,  "/scratch/palle.a/AirKeyboard/data/data_tensors/samples");
    saveSample(0.005, 0.05,  "/scratch/palle.a/AirKeyboard/data/data_tensors/full_data");

    
    torch::manual_seed(0);
    torch::cuda::manual_seed(0);

    auto data = loadSamples("/scratch/palle.a/AirKeyboard/data/data_tensors/full_data");
    Dataset train = data[0];
    Dataset test = data[1];

    // train = train.sample(0.1);
    // test = test.sample(0.5);

    cout << "running Training!" << endl;

    trainModel(train, test, true,1,  "model_full_IOU.pt");
    //trainModel(train, test, true,1, "sample_MSE.pt");

    evaluate(test, true,"model_full_IOU.pt");
    //evaluate(test, true,"sample_MSE.pt");
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