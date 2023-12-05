#include <torch/torch.h>
#include <iostream>
#include "train.h"
#include "data_work.h"
#include <any>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>  // Include xrandom header for randn function
using namespace std;

// Assuming your model takes an input of shape (n, 21, 128, 128)
torch::Tensor generateRandomInput(int n, int channels, int height, int width) {
    return torch::randn({n, channels, height, width});
}

// Assuming your model returns predictions of the same shape as the input
torch::Tensor yourModel(const torch::Tensor& input) {
    // Replace this with your actual model inference code
    return input * 2;  // This is just a placeholder, replace with your model
}

float calculateMSE(const torch::Tensor& predictions, const torch::Tensor& targets) {
    torch::Tensor mse = torch::mse_loss(predictions, targets);
    return mse.item<float>();
}
void xTensorTest(){
        // Create a random 2D array with dimensions 3x4
    xt::xarray<double> array = xt::random::randn<double>({3, 4});

    // Print the original array
    std::cout << "Original Array:\n" << array << std::endl;

    // Perform operations on the array (e.g., element-wise multiplication)
    array = array * 2.0;

    // Print the modified array
    std::cout << "Modified Array (element-wise multiplication by 2.0):\n" << array << std::endl;

    // Accessing elements
    double element = array(1, 2);
    std::cout << "Element at (1, 2): " << element << std::endl;

}

void testMSEWithDifferentInputSizes() {
    // Set common parameters
    int channels = 21;
    int height = 128;
    int width = 128;

    // Varying batch sizes for testing
    std::vector<int> batchSizes = {5, 10, 20, 30};

    for (int batchSize : batchSizes) {
        // Generate random input data
        torch::Tensor input = generateRandomInput(batchSize, channels, height, width);

        // Get model predictions
        torch::Tensor predictions = yourModel(input);

        // Assuming you have ground truth targets (replace with actual targets)
        torch::Tensor targets = input;//generateRandomInput(batchSize, channels, height, width);

        // Calculate MSE
        float mse = calculateMSE(predictions, targets);

        // Print the results
        std::cout << "Batch Size: " << batchSize << ", MSE: " << mse << std::endl;
    }
}


// Function to verify differentiability of the loss
void verifyDifferentiability(const torch::Tensor& input, const torch::Tensor& target) {

    // std::vector<T> losses;
    // losses.push_back(IouLoss());
    // losses.push_back(MSELoss());

        // Set requires_grad to true for input and target
        torch::Tensor inputWithGrad = input.clone().detach().requires_grad_(true);
        torch::Tensor targetWithGrad = target.clone().detach().requires_grad_(true);

        // Create the IouLoss module
        IouLoss iouLoss;
    // MSELoss mseLoss

        // Compute the IOU loss
        torch::Tensor loss = iouLoss.forward(inputWithGrad, targetWithGrad);
        //torch::Tensor mloss = mseLoss.forward(inputWithGrad, targetWithGrad);

        // Perform a backward pass
        loss.backward();
        //mloss.backward();

        // Check if gradients are computed successfully
        if (inputWithGrad.grad().defined() && targetWithGrad.grad().defined()) {
            std::cout << "Gradients computed successfully." << std::endl;
        } else {
            std::cout << "Error: Gradients not computed." << std::endl;
        }
    
}

void testIOULoss(){
    for (int i= 0; i < 2; i++){
    auto mse = MSELoss();
    auto iou = IouLoss();

    // Create a tensor of shape (n, 21, 128, 128)
    torch::Tensor input = torch::randn({5, 21, 128, 128});

    // Create a tensor of shape (n, 21, 128, 128)
    torch::Tensor target = input;

    torch::Tensor iou_loss  = iou.forward(input,target);
    torch::Tensor mse_loss  = mse.forward(input,target);
    torch::Tensor fmse = torch::mse_loss(input, target);
    cout << "IOU of identity : " << iou_loss.item<float>() << endl;
    cout << "MSE of identity : " << mse_loss.item<float>() << endl;
    cout << "func MSE of identity : " << fmse.item<float>() << endl;
    verifyDifferentiability(input,target);




    target = torch::randn({5, 21, 128, 128});
    iou_loss  = iou.forward(input,target);
    mse_loss  = mse.forward(input, target);
    fmse = torch::mse_loss(input, target);

    cout << "IOU of random : " << iou_loss.item<float>() << endl;
    cout << "MSE of random: " << mse_loss.item<float>() << endl;
    cout << "func MSE of random : " << fmse.item<float>() << endl;
    verifyDifferentiability(input,target);


    input = torch::zeros({1, 1, 128, 128});
    target = torch::zeros({1, 1, 128, 128});

    input[0][0][10][10] = 1;
    target[0][0][10][10] = 1;

    iou_loss  = iou.forward(input,target);
    mse_loss  = mse.forward(input, target);
    fmse = torch::mse_loss(input, target);


    cout << "IOU of perfect match : " << iou_loss.item<float>() << endl;
    cout << "MSE of perfect match: " << mse_loss.item<float>() << endl;
    cout << "func MSE of perfect match : " << fmse.item<float>() << endl;
    verifyDifferentiability(input,target);

    input = torch::zeros({1, 1, 128, 128});
    target = torch::zeros({1, 1, 128, 128});

    input[0][0][10][10] = 1;
    target[0][0][10][10] = 1;
    input[0][0][10][11] = 1;
    target[0][0][10][9] = 1;

    iou_loss  = iou.forward(input,target);
    mse_loss  = mse.forward(input, target);
    fmse = torch::mse_loss(input, target);


    cout << "IOU of 1/3: " << iou_loss.item<float>() << endl;
    cout << "MSE of 1/3: " << mse_loss.item<float>() << endl;
    cout << "func MSE of 1/3 : " << fmse.item<float>() << endl;
    verifyDifferentiability(input,target);


    }

    return;

}
template <typename LossType>
void testSpecificLoss(LossType loss){

    // Create a tensor of shape (n, 21, 128, 128)
    torch::Tensor input = torch::randn({5, 21, 128, 128});

    // Create a tensor of shape (n, 21, 128, 128)
    torch::Tensor target = torch::randn({5, 21, 128, 128});
    torch::Tensor lossVal  = loss.forward(input,target);
  
    cout << "Loss of random : " << lossVal.item<float>() << endl;

    

}


int main() {
    // Run the function to test MSE with different input sizes
    //testMSEWithDifferentInputSizes();
    testIOULoss();
    cout << "Testing specific loss "<< endl;
    IouLoss loss_func;
    testSpecificLoss(IouLoss());
    testSpecificLoss(MSELoss());
    return 0;
}