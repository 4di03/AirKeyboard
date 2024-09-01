#include <torch/torch.h>
#include <vector>
#pragma once
using namespace std;

template <typename T>
void printVector(const std::vector<T>& vec) {
    for (const auto& el : vec) {
        std::cout << el << ", "; 
    }
    std::cout << std::endl;
}

template <typename T>
void printType(const T& object) {
    std::cout << typeid(object).name() << std::endl;
}



std::string getTensorString(torch::Tensor tensor);
void printTensor(const torch::Tensor tensor);