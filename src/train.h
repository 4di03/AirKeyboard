#include <torch/torch.h>
#include <cstdlib>
#include "data_work.h"
#include "utils.h"
#include "model.h"
#include <stdexcept>
#pragma once // or use include guards
using namespace std;


class Loss : public torch::nn::Module {
public:
    virtual torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& target) = 0;
    virtual std::string getName() = 0;
    //~Loss() = 0;
};

class MSELoss : public Loss{
public:
torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& target) {
    return torch::mse_loss(input ,target);
}
std::string getName(){ return "MSE";}

};




class IouLoss : public Loss{
private:
float eps = 1e-12;

torch::Tensor opSum(torch::Tensor t){
    return t.sum(-1).sum(-1); // sums along last two axis (n,m,a,b) -> (n,m)
}

public:


torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& target) {
    // IOU = (Intersection of Union) / (Union of Union)
    // Loss = 1 - IOU

    // Ensure input and target are of the same shape
    if (input.sizes() != target.sizes()) {
        std::ostringstream error_message;
        error_message << "Input and target must have the same shape. "
                        << "Input shape: " << input.sizes() 
                        << ", Target shape: " << target.sizes();
        throw std::invalid_argument(error_message.str());
    }

    auto I = opSum(input * target);

    auto U = opSum(torch::pow(input,2)) + opSum(torch::pow(target,2)) - I;
    // Add a small epsilon to avoid division by zero

    auto iou = (I +  eps) / (U + eps);
    // shape of iou is (n_samples, n_channels)
    iou = torch::mean(iou);

    auto loss = 1.0 - iou;

    return loss;
}
std::string getName(){ return "IOU";}
};

class TrainParams {
public:
    Loss* loss_fn;
    ModelBuilder* modelBuilder;
    //int initNeurons = 16;
    float batchSize = 64;
    int nEpochs = 100;
    //int levels = 4;
    bool cuda = true;
    float propDataUsed = 1;
    std::string model_name = "default_model.pt";
    float propVal = 0.1;
    bool standardize = true;
    std::string modelPath = "";
    nlohmann::json modelParams;
   // builder for trian parameter
    TrainParams(Loss* lossFn =nullptr, ModelBuilder* mb = nullptr){
        this->loss_fn = lossFn;
        this->modelBuilder = mb;

    }


    TrainParams setPropVal(float propVal){
        this->propVal = propVal;
        return *this;
    }

    TrainParams setModelPath(std::string modelPath){
        this->modelPath = modelPath;
        return *this;
    }

    TrainParams  setLoss(Loss* loss_fn) {
        this->loss_fn = loss_fn;
        return *this;
    }

    TrainParams setStandardize(bool standardize){
        this->standardize = standardize;
        return *this;
    }

    // TrainParams setNeurons(int n) {
    //     this->initNeurons = n;
    //     return *this;
    // }

    TrainParams  setBatchSize(float batchSize) {
        this->batchSize = batchSize;
        return *this;
    }

    TrainParams setModelParams(nlohmann::json modelParams){
        this->modelParams = modelParams;
        return *this;
    }
    TrainParams  setEpochs(int n) {
        this->nEpochs = n;
        return *this;
    }

    // TrainParams setLevels(int levels) {
    //     this->levels = levels;
    //     return *this;
    // }

    TrainParams  setCuda(bool cuda) {
        this->cuda = cuda;
        return *this;
    }

    TrainParams  setPropDataUsed(float propDataUsed) {
        this->propDataUsed = propDataUsed;
        return *this;
    }

    TrainParams setModelName(const std::string& modelName) {
        this->model_name = modelName;
        this->setModelPath(getModelPath(modelName, "final_model.pt"));
        return *this;
    }

};


void trainModel(Dataset& train, 
                Dataset& test, 
                TrainParams tp);

void evaluate(Dataset& test, TrainParams tp, std::string saveName, bool draw);

Loss* getLoss(std::string lossName);
