
#include <iostream>
#include <opencv2/opencv.hpp>
//#include <torch/torch.h>
#include <ctime>


int main() {
    // Open a connection to the webcam (usually 0 for the default webcam)
    cv::VideoCapture cap(0);

    std::cout << "Expected FPS: " << cap.get(cv::CAP_PROP_FPS) << std::endl;
    int ct = 0;


    // torch::Tensor tensor = torch::rand({2, 3});
    // std::cout << tensor << std::endl;

    // Check if the webcam opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam." << std::endl;
        return -1;
    }

    // Create a window to display the webcam feed
    cv::namedWindow("Webcam Feed", cv::WINDOW_NORMAL);

    // get start time 
    std::time_t result = std::time(nullptr);


    // Main loop
    while (true) {
        ct++;
        // Capture a frame from the webcam
        cv::Mat frame;
        cap >> frame;

        // Check if the frame is empty
        if (frame.empty()) {
            std::cerr << "Error: Webcam disconnected or reached end of stream." << std::endl;
            break;
        }

        std::time_t cur_time = std::time(nullptr);
    

        float time_diff = cur_time - result;
        std::cout << "time diff: " << time_diff << std::endl;
        float avg_fps = ct/ time_diff;

        
        // Display the frame in the window
        cv::imshow("Webcam Feed", frame);

        std::cout << "Current FPS: " << avg_fps << std::endl;

        // Break the loop if the user presses the 'ESC' key
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // Release the webcam and close the window
    cap.release();
    cv::destroyAllWindows();

    return 0;
}