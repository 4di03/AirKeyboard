#include <torch/torch.h>
#include "data_work.h"

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


class MSELoss : public torch::nn::Module{
public:
torch::Tensor forward(const torch::Tensor input, const torch::Tensor target) {
    return torch::mse_loss(input ,target);
}


};

class IouLoss : public torch::nn::Module {
private:
float eps = 1e-12;
public:

torch::Tensor forward(const torch::Tensor input, const torch::Tensor target) {
    // IOU = (interesting Area/ Union Area) of joint regions
    // Loss = 1- IOU
    //Loss is calculated for each heatmap separately, then averaged among all 21 heatmaps, and then averaged among the images in the batc

    auto I = input * target;

    I =  I.sum({2,3}); // sum along last two axes to get intersection value per heatmap


    auto U = input  + target - input * target; // A OR  B - (A and B)


    U = U.sum({2,3}); // sum along last two axes to get  union value per heatmap, add epsilon to prevent 0 divison

    auto iou = (I+eps)/(U+eps);

    auto iou_loss =1- iou;

    iou_loss = iou_loss.mean(1);// average IOU across 21 heatmaps, to get average IOU per sample


    auto loss =  iou_loss.mean(0);// get average across all samples in batch


    return loss;


}
};


void trainModel(Dataset train, 
                Dataset test, 
                bool cuda , 
                float propDataUsed ,
                std::string model_name );

void evaluate(Dataset test, bool cuda, std::string model_name);

