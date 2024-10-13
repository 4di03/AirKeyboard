#include <vector>
#include <typeinfo>
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>
#include "utils.h"
#include "constants.h"
using namespace std;




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

std::string getModelPath(std::string model_name){

    return std::string(DATA_PATH) + "/models/" + model_name +"/final_model.pt";

}