#include <vector>
#include <typeinfo>
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <fstream>            // For std::ofstream
#include <filesystem>
#include "utils.h"
#include "constants.h"
#include "json.hpp"
using namespace std;



// Function to count the number of NaN values in a torch::Tensor
int countNaNs(const torch::Tensor& tensor) {
    // Use torch::isnan to create a mask where NaN elements are true (1) and others are false (0)
    torch::Tensor nan_mask = torch::isnan(tensor);
    
    // Sum the mask to get the total number of NaN values
    int nan_count = nan_mask.sum().item<int>();
    
    return nan_count;
}
std::string getTensorString(torch::Tensor tensor){
    std::ostringstream stream;
    stream << tensor;
    std::string tensor_string = stream.str();

    return tensor_string;
}

void printTensor(const torch::Tensor tensor){

    std::string tensor_string = getTensorString(tensor);
    std::cout << tensor_string << std::endl;

}

/*
Gets the current timestamp in form MMDDYYYY-HHmm 
*/
std::string timestampAsString(){
    // Get the current time
    std::time_t t = std::time(nullptr);
    std::tm* now = std::localtime(&t);

    // Create a stringstream to format the timestamp
    std::stringstream ss;
    
    // Format the timestamp as MMDDYYYY-HHmm
    ss << std::setfill('0') << std::setw(2) << (now->tm_mon + 1)  // Month (tm_mon is 0-based, so we add 1)
       << std::setw(2) << now->tm_mday   // Day
       << std::setw(4) << (now->tm_year + 1900)  // Year (tm_year is years since 1900)
       << "-" << std::setw(2) << now->tm_hour  // Hour
       << std::setw(2) << now->tm_min;  // Minutes

    return ss.str();

}

/**
* Makes the directory at the filepath.
*/
void createDirectory(const std::string dirPath) {
    try {
        // Check if the directory already exists
        if (!std::filesystem::exists(dirPath)) {
            // Create the directory
            if (!std::filesystem::create_directories(dirPath)) {
                std::cerr << "Failed to create directory: " << dirPath << std::endl;
            }
        } 
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
}

std::string getDirectoryName(const std::string& fileName) {
    try {
        // Create a path object from the filename
        std::filesystem::path filePath(fileName);
        
        // Get the parent directory path
        std::filesystem::path directory = filePath.parent_path();
        
        return directory.string();
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
        return "";
    }
}

/**
Gets file path to save model. The directory is named after modelName, and the file is named as fileName.
you can pass a file wihtin a nested diretory to filename to save it within that.
*/
std::string getModelPath(std::string modelName, std::string fileName){
    return std::string(DATA_PATH) + "/models/" + modelName +"/" + fileName;

}


void saveJsonToFile(nlohmann::json jsonObj, std::string jsonSavePath) {
    try {
        // Open a file stream to the specified path
        std::ofstream file(jsonSavePath);

        // Check if the file stream is open
        if (!file.is_open()) {
            throw std::ios_base::failure("Failed to open file: " + jsonSavePath);
        }

        // Write the JSON object to the file with proper formatting
        file << jsonObj.dump(4);  // '4' adds indentation for readability

        // Close the file stream
        file.close();

        std::cout << "JSON saved to " << jsonSavePath << " successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error saving JSON to file: " << e.what() << std::endl;
    }
}