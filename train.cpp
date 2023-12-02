#include <iostream>
#include <torch/torch.h>
#include "data_work.h"
#include <cmath>
#include <vector>
#include <torch/script.h>

using namespace std;


// Function to print the device of a PyTorch tensor
void printTensorDevice(const torch::Tensor& tensor) {
    // Get the device of the tensor
    torch::Device device = tensor.device();

    cout << device << endl;
    // Print the device type (CPU or CUDA)
    if (device.is_cpu()) {
        std::cout << "Tensor is on CPU." << std::endl;
    } else if (device.is_cuda()) {
        std::cout << "Tensor is on GPU." << std::endl;
    } else {
        std::cout << "Tensor device is unknown." << std::endl;
    }
}
// Function to print the device of a PyTorch module
void printModuleDevice(const torch::nn::Module& module) {
    // Get the device of the module parameters
    for (const auto& parameter : module.parameters()) {
        torch::Device device = parameter.device();

        // Print the device type (CPU or CUDA)
        if (device.is_cpu()) {
            std::cout << "Module is on CPU." << std::endl;
            return;  // Only need to print once
        } else if (device.is_cuda()) {
            std::cout << "Module is on GPU." << std::endl;
            return;  // Only need to print once
        }
    }

    // If no parameters found, assume the module is on CPU
    std::cout << "Module is on CPU." << std::endl;
}


class ConvBlock : public torch::nn::Module {
private:
    torch::nn::BatchNorm2d batchNorm1 = nullptr, batchNorm2 = nullptr;
    torch::nn::Conv2d conv1 = nullptr, conv2 = nullptr;
    torch::nn::Functional relu1 = nullptr, relu2 = nullptr;
    bool debug;
public:
    torch::nn::Sequential block = nullptr;

//    torch::nn::Sequential block;



    ConvBlock(int inChannels = 3, int outChannels = 3, int kernelSize = 3, int padding = 1, bool debug = false) : debug(debug)
    {
        /**
         * @brief This function creates a convolutional block
         * @param inChannels: int
         * @param outChannels: int
         * @param kernelSize: int
         * @param padding: int
         * @return torch::nn::Sequential
         */
        // Define each part of the block separately
        batchNorm1 = register_module("batchNorm1", torch::nn::BatchNorm2d(inChannels));
        conv1 = register_module("conv1", torch::nn::Conv2d(
                torch::nn::Conv2dOptions(inChannels, outChannels, kernelSize).padding(padding)));
        relu1 = register_module("relu1", torch::nn::Functional(torch::relu));

        batchNorm2 = register_module("batchNorm2", torch::nn::BatchNorm2d(outChannels));
        conv2 = register_module("conv2", torch::nn::Conv2d(
                torch::nn::Conv2dOptions(outChannels, outChannels, kernelSize).padding(padding)));
        relu2 = register_module("relu2", torch::nn::Functional(torch::relu));
        debug = debug;

        // Define the block as a sequence of the above parts
        block = torch::nn::Sequential(
                batchNorm1,
                conv1,
                relu1,
                batchNorm2,
                conv2,
                relu2
        );


    }

    torch::Tensor forward(torch::Tensor x) {
        //cout << "in CONV BLOCK forward" << debug <<  endl;
        if (debug) {
            std::cout << "Input: " << x.sizes() << std::endl;

            // Store the output of each layer for debugging
            torch::Tensor out1 = batchNorm1->forward(x);
            std::cout << "BatchNorm2d_1: " << out1.sizes() << std::endl;

            torch::Tensor out2 = conv1->forward(out1);
            std::cout << "Conv2d_1: " << out2.sizes() << std::endl;

            torch::Tensor out3 = relu1->forward(out2);
            std::cout << "Functional_1: " << out3.sizes() << std::endl;

            torch::Tensor out4 = batchNorm2->forward(out3);
            std::cout << "BatchNorm2d_2: " << out4.sizes() << std::endl;

            torch::Tensor out5 = conv2->forward(out4);
            std::cout << "Conv2d_2: " << out5.sizes() << std::endl;
            torch::Tensor out6 = relu2->forward(out5);

            std::cout << "Functional_2: " << out6.sizes() << std::endl;

            return out6;

        } else {
            return block->forward(x);
        }

    }

};


class UNet : public torch::nn::Module {
public:
    torch::nn::Sequential model;

    UNet(int nChannels, int initNeurons) {
        int c = nChannels;

        model = torch::nn::Sequential(
                ConvBlock(c, initNeurons),
                ConvBlock(initNeurons, initNeurons * 2),
                ConvBlock(initNeurons * 2, initNeurons * 4),
                ConvBlock(initNeurons * 4, initNeurons * 8),
                ConvBlock(initNeurons * 12, initNeurons * 4),
                ConvBlock(initNeurons * 6, initNeurons * 2),
                ConvBlock(initNeurons * 3, initNeurons),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(initNeurons, 21, 1)),
                torch::nn::Functional(torch::sigmoid)
        );


    }


    torch::Tensor forward(torch::Tensor x) {
        return model->forward(x);
    }
};

class ResidualUNet : public torch::nn::Module {
private:
    std::shared_ptr<ConvBlock> conv1, conv2, conv3, conv4, conv5, conv6, conv7;
    torch::nn::ConvTranspose2d upsample1 = nullptr;
    torch::nn::ConvTranspose2d upsample2 = nullptr;
    torch::nn::ConvTranspose2d upsample3 =  nullptr;
    torch::nn::Conv2d finalConv = nullptr;
    torch::nn::Functional sigmoid = nullptr;
    torch::nn::Sequential model;

    bool debug;
public:

    ResidualUNet(int nChannels = 3, int initNeurons = 16, bool debug = false) : debug(debug)
    {

        conv1 = register_module<ConvBlock>("conv1", std::make_shared<ConvBlock>(nChannels, initNeurons));
        conv2 = register_module<ConvBlock>("conv2", std::make_shared<ConvBlock>(initNeurons, initNeurons * 2));
        conv3 = register_module<ConvBlock>("conv3", std::make_shared<ConvBlock>(initNeurons * 2, initNeurons * 4));
        conv4 = register_module<ConvBlock>("conv4", std::make_shared<ConvBlock>(initNeurons * 4, initNeurons * 8));
        conv5 = register_module<ConvBlock>("conv5", std::make_shared<ConvBlock>(initNeurons * 12, initNeurons * 4));
        conv6 = register_module<ConvBlock>("conv6", std::make_shared<ConvBlock>(initNeurons * 6, initNeurons * 2));
        conv7 = register_module<ConvBlock>("conv7", std::make_shared<ConvBlock>(initNeurons * 3, initNeurons));

//        // Define residual connections (skip connections)
//        upsample1 = register_module("upsample1", torch::nn::ConvTranspose2d(
//                torch::nn::ConvTranspose2dOptions(initNeurons, initNeurons, 2).stride(2)));
//        upsample2 = register_module("upsample2", torch::nn::ConvTranspose2d(
//                torch::nn::ConvTranspose2dOptions(initNeurons * 2, initNeurons * 2, 2).stride(2)));
//        upsample3 = register_module("upsample3", torch::nn::ConvTranspose2d(
//                torch::nn::ConvTranspose2dOptions(initNeurons * 4, initNeurons * 4, 2).stride(2)));

        finalConv = register_module("finalConv", torch::nn::Conv2d(torch::nn::Conv2dOptions(initNeurons, 21, 1)));
        sigmoid = register_module("sigmoid", torch::nn::Functional(torch::sigmoid));

    }


    torch::Tensor forward(torch::Tensor x) {
        if (debug) {

            cout << "x INPUT: " << x.sizes() << endl;

            torch::Tensor conv1Out = conv1->forward(x);
            cout << "conv1Out: " << conv1Out.sizes() << endl;


//            torch::Tensor upsampled1 = upsample1->forward(conv1Out);
//            cout << "upsampled1: " << upsampled1.sizes() << endl;

            torch::Tensor conv2Out = conv2->forward(conv1Out);
            cout << "conv2Out: " << conv2Out.sizes() << endl;
//
//            torch::Tensor upsampled2 = upsample2->forward(conv2Out);
//            cout << "upsampled2: " << upsampled2.sizes() << endl;

            torch::Tensor conv3Out = conv3->forward(conv2Out);
            cout << "conv3Out: " << conv3Out.sizes() << endl;

//            torch::Tensor upsampled3 = upsample3->forward(conv3Out);
//            cout << "upsampled3: " << upsampled3.sizes() << endl;

            torch::Tensor conv4Out = conv4->forward(conv3Out);
            cout << "conv4Out: " << conv4Out.sizes() << endl;


            torch::Tensor skipConnected1 = torch::cat({conv3Out,conv4Out},1);  // Concat along columns axis (3rd dimensions) conv4Out + upsampled3;
            cout << "skipConnected1: " << skipConnected1.sizes() << endl;

            torch::Tensor conv5Out = conv5->forward(skipConnected1);


            cout << "conv5Out: " << conv5Out.sizes() << endl;

            torch::Tensor skipConnected2 = torch::cat({conv2Out,conv5Out }, 1);
            cout << "skipConnected2: " << skipConnected2.sizes() << endl;


            torch::Tensor conv6Out = conv6->forward(skipConnected2);
            cout << "conv6Out: " << conv6Out.sizes() << endl;

            torch::Tensor skipConnected3 = torch::cat({conv1Out, conv6Out },1);
            cout << "skipConnected3: " << skipConnected3.sizes() << endl;


            torch::Tensor conv7Out = conv7->forward(skipConnected3);
            cout << "conv7Out: " << conv7Out.sizes() << endl;



            torch::Tensor finalOut = finalConv->forward(conv7Out);
            cout << "finalOut: " << finalOut.sizes() << endl;


            return sigmoid->forward(finalOut);
        } else{

            torch::Tensor conv1Out = conv1->forward(x);

            torch::Tensor conv2Out = conv2->forward(conv1Out);

            torch::Tensor conv3Out = conv3->forward(conv2Out);

            torch::Tensor conv4Out = conv4->forward(conv3Out);

            torch::Tensor skipConnected1 = torch::cat({conv3Out,conv4Out},1);  // Concat along columns axis (3rd dimensions) conv4Out + upsampled3;

            torch::Tensor conv5Out = conv5->forward(skipConnected1);

            torch::Tensor skipConnected2 = torch::cat({conv2Out,conv5Out }, 1);

            torch::Tensor conv6Out = conv6->forward(skipConnected2);

            torch::Tensor skipConnected3 = torch::cat({conv1Out, conv6Out },1);

            torch::Tensor conv7Out = conv7->forward(skipConnected3);

            torch::Tensor finalOut = finalConv->forward(conv7Out);

            return sigmoid->forward(finalOut);


        }
    }


};

torch::Tensor removeExtraDim(torch::Tensor x){
    std::vector<int64_t> sizes;
    
    for (int64_t size : x.sizes()){
        if (size != 1){
            sizes.push_back(size);
        }
    }

    return x.view(sizes);
}

float getMean(std::vector<float> nums){
    float sum = 0;

    for (float n :nums){
        sum += n;
    }

    return sum/nums.size();

}


torch::Device initDevice(bool cuda){
    torch::Device device(torch::kCUDA,0);
    if (!cuda){
        device = torch::Device(torch::kCPU);
    }else{
        cout <<"USING CUDA" << endl;
    }

    return device;
}


// Load Model
void loadModel(const std::string& model_path, torch::nn::Module model) {
    //torch::jit::script::Module model;

    try {
        torch::serialize::InputArchive input_archive;
        input_archive.load_from(model_path);
        model.load(input_archive);
        //model = torch::jit::load(model_path);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
    }

}


void evaluateTest( Dataset test, bool cuda, ResidualUNet model){
    torch::Device device = initDevice(cuda);

    model.to(device);

    test.sample(0.025);
    test.y = removeExtraDim(test.y);


    cout << test.x.sizes() << test.y.sizes() << endl;

    auto test_x = test.x.to(device);
    auto test_y = test.y.to(device);


    model.eval();
    auto pred = model.forward(test_x);

    cout << pred.sizes() << endl;

    torch::Tensor loss = torch::mse_loss(pred, test_y);

    cout <<"Test Loss: " << loss.item<float>() << endl;

}

void evaluate(Dataset test, bool cuda = true){

    ResidualUNet model;

    loadModel("/scratch/palle.a/AirKeyboard/data/models/sample_model.pt", model);

    evaluateTest(test,cuda,model);

    // torch::Device device = initDevice(cuda);

    // model.to(device);

    // test.sample(0.025);
    // test.y = removeExtraDim(test.y);


    // cout << test.x.sizes() << test.y.sizes() << endl;

    // auto test_x = test.x.to(device);
    // auto test_y = test.y.to(device);


    // model.eval();

    // auto pred = model.forward(test_x);

    // cout << pred.sizes() << endl;

    // torch::Tensor loss = torch::mse_loss(pred, test_y);

    // cout <<"Test Loss: " << loss.item<float>() << endl;




    //cout << model << endl;
}

void trainModel(Dataset train, Dataset test, bool cuda = true, float propDataUsed = 0.3) {
    /**
     * @brief This function trains the model
     * @param train: Dataset
     * @param test: Dataset
     * ASSUMES CUDA WITH GPU IS AVAILABLE.
     */


    train.sample(propDataUsed);
    test.sample(propDataUsed);


    cout << "train x shape: " << train.x.sizes() << endl;
    cout << "train y shape: " << train.y.sizes() << endl;
    cout << "test x shape: " << test.x.sizes() << endl;
    cout << "test y shape: " << test.y.sizes() << endl;

    
    auto sizes = train.x.sizes();

    int c = sizes[1];
    int nTrainSamples = sizes[0];
    int initNeurons = 16;
    int batchSize = 128;
    int nEpochs = 40;
    ResidualUNet model = ResidualUNet(c, initNeurons);

    torch::Device device = initDevice(cuda);
    model.to(device);

    printModuleDevice(model);

    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.01));

    for (size_t epoch = 1; epoch <= nEpochs; ++epoch) {
        // Iterate over batches
        std::cout << "Epoch: " << epoch << std::endl;

        int nBatches = round(nTrainSamples / batchSize);
        std::vector<float> batchLosses;
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < nBatches; ++i) {
            // Get the batch
            auto batch = train.getBatch(i, batchSize);
            auto x = batch.first;
            auto y = batch.second;

    


            // Zero gradients
            optimizer.zero_grad();


           // cout << "x shape: " << x.sizes() << endl;
            // Forward pass
            x = x.to(device);

            torch::Tensor y_pred = model.forward(x);


            y = y.to(device);
            y = removeExtraDim(y); // to remove extra axis for 1 channel image

            // cout << "y_pred shape: " << y_pred.sizes() << endl;
            // cout << "y shape: " << y.sizes() << endl;
            // Compute Loss

            torch::Tensor loss = torch::mse_loss(y_pred, y);
            batchLosses.push_back(loss.item<float>());

            // Backward pass
            loss.backward();

            // Update the parameters
            optimizer.step();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        // Print the time taken for the forward pass
        std::cout << "Epoch took " << duration.count() << " microseconds." << std::endl;


        cout << "Epoch Mean Loss " << getMean(batchLosses) <<endl;


    }

    model.eval();

    cout << "Post-training loss:";
    evaluateTest(test, cuda, model);

    // SAVE MODEL
    string model_path = "/scratch/palle.a/AirKeyboard/data/models/sample_model.pt";
    torch::serialize::OutputArchive output_archive;
    model.save(output_archive);
    output_archive.save_to(model_path);
}



// torch::Tensor predict(torch::jit::script::Module model, torch::Tensor x){


//     std::vector<torch::jit::IValue> inputs;

//     inputs.push_back(x);

//     return model.forward(inputs).toTensor();
// }
