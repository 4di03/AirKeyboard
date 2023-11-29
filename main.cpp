
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <ctime>
#include "data_work.h"
#include "train.h"
using namespace std;


void saveSample(float propTrain = 0.1, float propTest = 0.1){
    /**
     * @brief This function  loads data and saves samples o a tensor file in data/data_tensors
     */
    std::string trainData = "/Users/adithyapalle/work/CS5100/Project/data/hanco_all/HanCo/train_keypoint_data.csv";
    Dataset train = prepData(trainData, propTrain);

    cout << "train x shape: " << train.x.sizes() << endl;
    cout << "train y shape: " << train.y.sizes() << endl;

    torch::Tensor xTrain = train.x;
    torch::Tensor yTrain = train.y;

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
    /**
     * @brief This function loads samples from the tensor files in data/data_tensors
     * @return std::tuple<Dataset>
     */
    torch::Tensor xTrain;
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
    Dataset test = {xTest, yTest};

    return {train, test};

}

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