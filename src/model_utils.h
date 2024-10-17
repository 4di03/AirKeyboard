#include <torch/torch.h>
#include <torch/script.h>
#pragma once // or use include guards

using namespace std;

void saveModel(const std::string& modelPath, torch::nn::Module& model);
void printTensorDevice(const torch::Tensor& tensor) ;
void printModuleDevice(const torch::nn::Module& module);
torch::Tensor standardizeImages(const torch::Tensor x, const torch::Tensor& means, const torch::Tensor& stds);
void printComputationGraphAndParams(const torch::jit::script::Module& model);
