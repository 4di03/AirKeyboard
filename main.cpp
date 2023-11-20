
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>


int main() {
    // Open a connection to the webcam (usually 0 for the default webcam)
    cv::VideoCapture cap(0);

    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;

    // Check if the webcam opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam." << std::endl;
        return -1;
    }

    // Create a window to display the webcam feed
    cv::namedWindow("Webcam Feed", cv::WINDOW_NORMAL);

    // Main loop
    while (true) {
        // Capture a frame from the webcam
        cv::Mat frame;
        cap >> frame;

        // Check if the frame is empty
        if (frame.empty()) {
            std::cerr << "Error: Webcam disconnected or reached end of stream." << std::endl;
            break;
        }

        // Display the frame in the window
        cv::imshow("Webcam Feed", frame);

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