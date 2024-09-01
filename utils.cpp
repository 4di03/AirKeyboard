#include <vector>
#include <typeinfo>
#include <torch/torch.h>
#include "utils.h"
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