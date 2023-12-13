<<<<<<< HEAD
#include "data_work.h"
void trainModel(Dataset train,  Dataset test);
=======
#include <torch/torch.h>
#include "data_work.h"
//#include "model.h"
#pragma once // or use include guards
using namespace std;
// class Loss : torch::nn::Module {
// public:
//     virtual torch::Tensor forward(const torch::Tensor input, const torch::Tensor target) = 0;
// };

// template <typename LossType>
// class LossComp {
// public:
//     LossType l;
//     LossComp(LossType l){
//         l = l;
//     }
//     LossType getLoss(){
//         return l;
//     }
// };


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

   // cout << " L52, inp: " << torch::max(input) << " target:" << torch::max(target) << endl;

    auto I = opSum(input * target);

    //cout << " L56 " << torch::max(I) << endl;

    auto U = opSum(torch::pow(input,2)) + opSum(torch::pow(target,2)) - I;
   // cout << " L57 " << torch::max(U) << endl;
    // Add a small epsilon to avoid division by zero

    
    auto iou = (I +  eps) / (U + eps);


  //  cout << " L58 " << torch::max(iou) << endl;

    iou = torch::mean(iou);

   // cout << " L59 " << torch::max(iou) << endl;


    auto loss = 1.0 - iou;
  //   cout << " L70 " << torch::max(loss) << endl;


    return loss;
}
std::string getName(){ return "IOU";}
};

class TrainParams {
public:
    Loss* loss_fn;
    int initNeurons = 16;
    float batchSize = 64;
    int nEpochs = 100;
    int levels = 4;
    bool cuda = true;
    float propDataUsed = 1;
    std::string model_name = "default_model.pt";
    float propVal = 0.1;
    bool standardize = true;
    std::string modelPath = "";
   // builder for trian parameter
    TrainParams(){

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

    TrainParams setNeurons(int n) {
        this->initNeurons = n;
        return *this;
    }

    TrainParams  setBatchSize(float batchSize) {
        this->batchSize = batchSize;
        return *this;
    }

    TrainParams  setEpochs(int n) {
        this->nEpochs = n;
        return *this;
    }

    TrainParams setLevels(int levels) {
        this->levels = levels;
        return *this;
    }

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
        return *this;
    }

    bool pretrainedModelReady(){
        return this->modelPath != "";
    }
};


void trainModel(Dataset& train, 
                Dataset& test, 
                TrainParams tp);

void evaluate(Dataset& test, TrainParams tp, bool draw);


//float evaluateTest( Dataset test, torch::Device device, Model& model, Loss& loss_fn);
>>>>>>> 9faf08865b02c831689cb0fbc1434c782d8b2966
