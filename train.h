#include <torch/torch.h>
#include "data_work.h"
#pragma once // or use include guards

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
public:


torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& target) {
    // IOU = (Intersection of Union) / (Union of Union)
    // Loss = 1 - IOU

    auto I = input * target;
    auto U = torch::pow(input,2) + torch::pow(target,2) - I;

    // Add a small epsilon to avoid division by zero
    auto iou = (I.sum() + eps) / (U.sum() + eps);

    // IoU loss
    auto loss = 1.0 - iou;

    return loss;
}
// torch::Tensor forward(const torch::Tensor input, const torch::Tensor target) {
//     // IOU = (interesting Area/ Union Area) of joint regions
//     // Loss = 1- IOU
//     //Loss is calculated for each heatmap separately, then averaged among all 21 heatmaps, and then averaged among the images in the batc
//     // input is (nx21x128,128)
//     //std::cout <<"L48" << input.sizes() << target.sizes() << std::endl;
//     auto I = input * target;

//     I =  I.sum({2,3}); // sum along last two axes to get intersection value per heatmap


//     auto U = (input*input).sum({2,3})  + (target*target).sum({2,3}) - (input * target).sum({2,3}); // A OR  B - (A and B)


// // U = U.sum({2,3}); // sum along last two axes to get  union value per heatmap, add epsilon to prevent 0 divison


//     auto iou = (I+eps)/(U+eps);

//     //std::cout << "L62 max iou: " << torch::max(iou).item<float>() <<  std::endl;


//     //auto iou_loss = 1- iou;

//     //std::cout << "L63 max 1-iou: " << torch::max(iou_loss).item<float>() <<  std::endl;


//     auto iou_loss = iou.mean(1);// average IOU across 21 heatmaps, to get average IOU per sample

//     //std::cout << "L64 max  hm-mean (iou_loss): " << torch::max(iou_loss).item<float>() <<  std::endl;

//     auto  meanIOU =  iou_loss.mean(0);// get average across all samples in batch

//     //std::cout << "L65 batch-mean (iou_loss): " <<  loss.item<float>() <<  std::endl;



//     return 1-meanIOU;


// }
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
   // builder for trian parameter
    TrainParams(){

    }


    TrainParams setPropVal(float propVal){
        this->propVal = propVal;
        return *this;
    }

    TrainParams  setLoss(Loss* loss_fn) {
        this->loss_fn = loss_fn;
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
};


void trainModel(Dataset& train, 
                Dataset& test, 
                TrainParams tp);

void evaluate(Dataset& test, TrainParams tp, bool draw);


