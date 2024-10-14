#include <torch/torch.h>
#include <torch/script.h>
#include "cunet/cunet.h"
#include "utils.h"
#include "model_utils.h"
#pragma once // or use include guards

using namespace std;

/**
* Defines a standardization preprocessing step in the model for the images 
*/
class StandardizeTransform {
public:
    StandardizeTransform() {
        // No initialization needed
        // won't applyign any transformation
    }
    StandardizeTransform(const torch::Tensor& means, const torch::Tensor& stds)
        : means_(means), stds_(stds) {}

    torch::Tensor transform(const torch::Tensor dset) const {
        // printTensorDevice(means_);
        // printTensorDevice(stds_);
        // printTensorDevice(dset);
        
        if (!means_.defined() || !stds_.defined()) {
          //  cout << "NOT STANDARDIZING!" <<std::endl;
            return dset;
        } else{
           //cout << "STANDARDIZING!" <<std::endl;
            return standardizeImages(dset, means_, stds_);
        }
        
    }

private:
    torch::Tensor means_;
    torch::Tensor stds_;
};


class Model : public torch::nn::Module {
public:
    StandardizeTransform transformer_;
    virtual torch::Tensor postNormForward(const torch::Tensor& x)=0;

    /**
    Forward pass with standardizatio preprocessing step.
    */
    torch::Tensor forward(const torch::Tensor& x) {
        torch::Tensor xClone;
        {
            // Apply the normalization to input image without tracking gradients
            torch::NoGradGuard no_grad;
            //cout <<"transforming" <<std::endl;
            xClone = transformer_.transform(x);
            //cout <<"post-transforming" <<std::endl;
        }
       return postNormForward(xClone);
       // return model->forward(x);
    }


    void setTransformer(StandardizeTransform transformer){
        transformer_ = transformer;
    }

    void to(torch::Device d){
       torch::nn::Module::to(d);    
    }


    void save(const std::string& modelPath){
        saveModel(modelPath, *this);
}
    
};

// Define a derived class using the provided torch::jit::script::Module
class JitModel : public Model {
public:
    torch::jit::script::Module model;

    // Constructor
    JitModel(std::string modelPath, torch::Device device){

        model = torch::jit::load(modelPath,device);


       cout << "L160: " <<std::endl;
        printComputationGraphAndParams(model);
        
    }

    torch::Tensor postNormForward(const torch::Tensor& x) override {
        // Forward pass using the loaded PyTorch JIT model
        // You may need to adjust the input tensor size and format based on your model
        // cout << "L135 "; 

        // printTensorDevice(x);
        auto ret =  model.forward({x}).toTensor();
        //cout <<"L136" <<std::endl;
        return ret ;
    }

       // Override the to method to move the model to the specified device
    void to(torch::Device d) {
        model.to(d);
    }



    void save(const std::string& modelPath){
        createDirectory(getDirectoryName(modelPath));
        cout << "performing jit save" <<std::endl;
        model.save(modelPath);
    }
};

class ConvBlock : public torch::nn::Module {
public:

    torch::nn::BatchNorm2d batchNorm1 = nullptr, batchNorm2 = nullptr;
    torch::nn::Conv2d conv1 = nullptr, conv2 = nullptr;
    torch::nn::Functional relu1 = nullptr, relu2 = nullptr;
    bool debug;
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
        //cout << "in CONV BLOCK forward" << debug << std::endl;
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


class ResidualUNet :public Model{
public:
    std::shared_ptr<ConvBlock> conv1, conv2, conv3, conv4, conv5, conv6, conv7;
    torch::nn::ConvTranspose2d upsample1 = nullptr;
    torch::nn::ConvTranspose2d upsample2 = nullptr;
    torch::nn::ConvTranspose2d upsample3 =  nullptr;
    torch::nn::Conv2d finalConv = nullptr;
    torch::nn::Functional sigmoid = nullptr;
    // torch::nn::Sequential model;
    torch::nn::MaxPool2d maxpool= nullptr;
    torch::nn::Upsample upsample = nullptr;


    bool debug;

    ResidualUNet(int nChannels = 3, int initNeurons = 16, int kernelSize = 3, bool debug = false) : debug(debug)
    {

        conv1 = register_module<ConvBlock>("conv1", std::make_shared<ConvBlock>(nChannels, initNeurons, kernelSize));
        conv2 = register_module<ConvBlock>("conv2", std::make_shared<ConvBlock>(initNeurons, initNeurons * 2,kernelSize));
        conv3 = register_module<ConvBlock>("conv3", std::make_shared<ConvBlock>(initNeurons * 2, initNeurons * 4, kernelSize));
        conv4 = register_module<ConvBlock>("conv4", std::make_shared<ConvBlock>(initNeurons * 4, initNeurons * 8, kernelSize));
        conv5 = register_module<ConvBlock>("conv5", std::make_shared<ConvBlock>(initNeurons * 12, initNeurons * 4, kernelSize));
        conv6 = register_module<ConvBlock>("conv6", std::make_shared<ConvBlock>(initNeurons * 6, initNeurons * 2, kernelSize));
        conv7 = register_module<ConvBlock>("conv7", std::make_shared<ConvBlock>(initNeurons * 3, initNeurons,kernelSize));

        finalConv = register_module("finalConv", torch::nn::Conv2d(torch::nn::Conv2dOptions(initNeurons, 21, 1).bias(false)));
        sigmoid = register_module("sigmoid", torch::nn::Functional(torch::sigmoid));
        maxpool = register_module("maxpool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));
        upsample = torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({2,2})).mode(torch::kBilinear).align_corners(false));

        cout <<"INIT RES-UNET" <<std::endl;

    }


    torch::Tensor  postNormForward(const torch::Tensor& x) {
        if (debug) {

            cout << "x INPUT: " << x.sizes() <<std::endl;

            torch::Tensor conv1Out = conv1->forward(x);
            cout << "conv1Out: " << conv1Out.sizes() <<std::endl;


            torch::Tensor conv2Out = conv2->forward(conv1Out);
            cout << "conv2Out: " << conv2Out.sizes() <<std::endl;


            torch::Tensor conv3Out = conv3->forward(conv2Out);
            cout << "conv3Out: " << conv3Out.sizes() <<std::endl;

            torch::Tensor conv4Out = conv4->forward(conv3Out);
            cout << "conv4Out: " << conv4Out.sizes() <<std::endl;


            torch::Tensor skipConnected1 = torch::cat({conv3Out,conv4Out},1);  // Concat along columns axis (3rd dimensions) conv4Out + upsampled3;
            cout << "skipConnected1: " << skipConnected1.sizes() <<std::endl;

            torch::Tensor conv5Out = conv5->forward(skipConnected1);


            cout << "conv5Out: " << conv5Out.sizes() <<std::endl;

            torch::Tensor skipConnected2 = torch::cat({conv2Out,conv5Out }, 1);
            cout << "skipConnected2: " << skipConnected2.sizes() <<std::endl;


            torch::Tensor conv6Out = conv6->forward(skipConnected2);
            cout << "conv6Out: " << conv6Out.sizes() <<std::endl;

            torch::Tensor skipConnected3 = torch::cat({conv1Out, conv6Out },1);
            cout << "skipConnected3: " << skipConnected3.sizes() <<std::endl;


            torch::Tensor conv7Out = conv7->forward(skipConnected3);
            cout << "conv7Out: " << conv7Out.sizes() <<std::endl;



            torch::Tensor finalOut = finalConv->forward(conv7Out);
            cout << "finalOut: " << finalOut.sizes() <<std::endl;

            auto sigm =sigmoid->forward(finalOut); 
            return sigm;
            // if (! normalize){
            //     return sigm;
            // }
            // // sigm is of shape (n , 21, 128 , 128)
            // torch::Tensor maxVals = sigm.sum({2,3});
            // maxVals = maxVals.view({sigm.sizes()[0], sigm.sizes()[1],1,1});
            // return sigm/(maxVals + 1e-12);// normalize output image 

        } else{

            torch::Tensor conv1Out = conv1->forward(x);

            torch::Tensor conv2Out = conv2->forward(maxpool->forward(conv1Out));

            torch::Tensor conv3Out = conv3->forward(maxpool->forward(conv2Out));

            torch::Tensor conv4Out = conv4->forward(maxpool->forward(conv3Out));

            torch::Tensor skipConnected1 = torch::cat({upsample(conv4Out),conv3Out},1);  // Concat along columns axis (3rd dimensions) conv4Out + upsampled3;

            torch::Tensor conv5Out = conv5->forward(skipConnected1);

            torch::Tensor skipConnected2 = torch::cat({upsample(conv5Out),conv2Out }, 1);

            torch::Tensor conv6Out = conv6->forward(skipConnected2);

            torch::Tensor skipConnected3 = torch::cat({upsample(conv6Out) ,conv1Out},1);

            torch::Tensor conv7Out = conv7->forward(skipConnected3);

            torch::Tensor finalOut = finalConv->forward(conv7Out);
            
            auto sigm =sigmoid->forward(finalOut); 

            return sigm;
            // if (! normalize){
            //     return sigm;
            // }
            // // sigm is of shape (n , 21, 128 , 128)
            // torch::Tensor maxVals = sigm.max({2,3});
            // maxVals = maxVals.view({sigm.sizes()[0], sigm.sizes()[1],1,1});
            // return sigm/(maxVals + 1e-12);// normalize output image 

        }
    }


};

class CuNet : public Model{
public:
    std::shared_ptr<CUNet2dImpl> model;

    CuNet(int inChannels = 3, int outChannels = 21, int initNeurons = 64, int levels = 4)
        : model(std::make_shared<CUNet2dImpl>(inChannels, outChannels, initNeurons, levels)) {
        model = register_module("model", model);
        cout <<"INIT CUNET" <<std::endl;

    }

 

    torch::Tensor  postNormForward(const torch::Tensor& x) {
 
       return model->forward(x);
    }


};


class ModelBuilder{
public:
virtual Model* build() =0; // used to initalize models consistently
};

class CuNetBuilder: public ModelBuilder{

public:
    int inChannels = 3;
    int outChannels = 21;
    int initNeurons = 64;
    int levels = 4;

    Model* build(){
        return new CuNet(this->inChannels, this->outChannels, this->initNeurons, this->levels);
    }
};
