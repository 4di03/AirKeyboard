#include <iostream>
#include <torch/torch.h>
#include "data_work.h"
#include <cmath>

class ConvBlock : public torch::nn::Module {
public:
    torch::nn::Sequential block;



    ConvBlock(int inChannels = 3,int outChannels =  3, int kernelSize = 3, int padding = 1){
        /**
         * @brief This function creates a convolutional block
         * @param inChannels: int
         * @param outChannels: int
         * @param kernelSize: int
         * @param padding: int
         * @return torch::nn::Sequential
         */
        block = torch::nn::Sequential(
                torch::nn::BatchNorm2d(inChannels),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, outChannels , kernelSize).padding(padding)),
                torch::nn::Functional(torch::relu),
                torch::nn::BatchNorm2d(outChannels),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(outChannels, outChannels, kernelSize).padding(padding)),
                torch::nn::Functional(torch::relu)
        );
    }

    torch::Tensor forward(torch::Tensor x){
        return block->forward(x);
    }

};


class UNet : public torch::nn::Module {
public:
    torch::nn::Sequential model;
    UNet(int nChannels, int initNeurons){
        int c = nChannels;

        model =  torch::nn::Sequential(
                ConvBlock(c, initNeurons),
                ConvBlock(initNeurons, initNeurons*2),
                ConvBlock(initNeurons*2, initNeurons*4),
                ConvBlock(initNeurons*4, initNeurons*8),
                ConvBlock(initNeurons*12, initNeurons*4),
                ConvBlock(initNeurons*6, initNeurons*2),
                ConvBlock(initNeurons*3, initNeurons),
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
    ConvBlock conv1, conv2, conv3, conv4, conv5, conv6, conv7;
    torch::nn::ConvTranspose2d upsample1 = NULL;
    torch::nn::ConvTranspose2d  upsample2 = NULL;
    torch::nn::ConvTranspose2d  upsample3 = NULL ;
    torch::nn::Conv2d finalConv = NULL;
    torch::nn::Functional sigmoid = NULL;
public:
    torch::nn::Sequential model;

    ResidualUNet(int nChannels, int initNeurons) {
//        conv1 = register_module("conv1", ConvBlock(nChannels, initNeurons));
//        conv2 = register_module("conv2", ConvBlock(initNeurons, initNeurons * 2));
//        conv3 = register_module("conv3", ConvBlock(initNeurons * 2, initNeurons * 4));
//        conv4 = register_module("conv4", ConvBlock(initNeurons * 4, initNeurons * 8));
//        conv5 = register_module("conv5", ConvBlock(initNeurons * 12, initNeurons * 4));
//        conv6 = register_module("conv6", ConvBlock(initNeurons * 6, initNeurons * 2));
//        conv7 = register_module("conv7", ConvBlock(initNeurons * 3, initNeurons));
//
//        // Define residual connections (skip connections)
//        upsample1 = register_module("upsample1", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(initNeurons , initNeurons , 2).stride(2)));
//        upsample2 = register_module("upsample2", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(initNeurons * 2, initNeurons * 2, 2).stride(2)));
//        upsample3 = register_module("upsample3", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(initNeurons * 4, initNeurons*4, 2).stride(2)));
//
//        finalConv = register_module("finalConv", torch::nn::Conv2d(torch::nn::Conv2dOptions(initNeurons, 21, 1)));
//        sigmoid = register_module("sigmoid", torch::nn::Functional(torch::sigmoid));



        conv1 = ConvBlock(nChannels, initNeurons);
        conv2 = ConvBlock(initNeurons, initNeurons * 2);
        conv3 = ConvBlock(initNeurons * 2, initNeurons * 4);
        conv4 = ConvBlock(initNeurons * 4, initNeurons * 8);
        conv5 = ConvBlock(initNeurons * 12, initNeurons * 4);
        conv6 = ConvBlock(initNeurons * 6, initNeurons * 2);
        conv7 = ConvBlock(initNeurons * 3, initNeurons);

        // Define residual connections (skip connections)
        upsample1 = register_module("upsample1", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(initNeurons , initNeurons , 2).stride(2)));
        upsample2 = register_module("upsample2", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(initNeurons * 2, initNeurons * 2, 2).stride(2)));
        upsample3 = register_module("upsample3", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(initNeurons * 4, initNeurons*4, 2).stride(2)));

        finalConv = register_module("finalConv", torch::nn::Conv2d(torch::nn::Conv2dOptions(initNeurons, 21, 1)));
        sigmoid = register_module("sigmoid", torch::nn::Functional(torch::sigmoid));

    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor conv1Out = conv1.forward(x);
        torch::Tensor conv2Out = conv2.forward(conv1Out);
        torch::Tensor conv3Out = conv3.forward(conv2Out);
        torch::Tensor conv4Out = conv4.forward(conv3Out);
        torch::Tensor conv5Out = conv5.forward(conv4Out);
        torch::Tensor conv6Out = conv6.forward(conv5Out);
        torch::Tensor conv7Out = conv7.forward(conv6Out);

        // Apply residual connections
        torch::Tensor upsampled1 = upsample1->forward(conv1Out);
        torch::Tensor upsampled2 = upsample2->forward(conv2Out);
        torch::Tensor upsampled3 = upsample3->forward(conv3Out);

        torch::Tensor skipConnected1 = conv7Out + upsampled1;
        torch::Tensor skipConnected2 = conv6Out + upsampled2;
        torch::Tensor skipConnected3 = conv5Out + upsampled3;

        torch::Tensor finalOut = finalConv->forward(skipConnected3);
        return sigmoid->forward(finalOut);
    }


};


void trainModel(Dataset train,  Dataset test){
    /**
     * @brief This function trains the model
     * @param train: Dataset
     * @param test: Dataset
     */

//    torch::nn::Sequential model(
//        torch::nn::Linear(2, 64),
//        torch::nn::Functional(torch::relu),
//        torch::nn::Linear(64, 64),
//        torch::nn::Functional(torch::relu),
//        torch::nn::Linear(64, 1)
//    );
    // initialize U-Net
    // Architecture:
    // Conv Block: BatchNorm2D -> Conv2D(in_channels =3 ,out_channels = N, size =3, padding = 1) -> ReLU -> BatchNorm2D -> Conv2D(in_channels = N, out_channels = N,size =3, padding = 1) -> ReLU -> MaxPool2D(size = 2, stride = 2)

    auto sizes =  train.x.sizes();
    int c = sizes[1];
    int nTrainSamples = sizes[0];
    int initNeurons = 16;
    int batchSize = 64;
    UNet model = UNet(c, initNeurons);

//    std::vector<torch::Tensor> trainTensors = {train.x,train.y};
//    auto train_loader = torch::data::make_data_loader(
//            torch::data::datasets::TensorDataset(trainTensors),
//            torch::data::DataLoaderOptions().batch_size(64).workers(2)
//    );

    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.01));

    for (size_t epoch = 1; epoch <= 100; ++epoch) {
        // Iterate over batches
        int nBatches = round(nTrainSamples/batchSize);
        for (int i = 0; i < nBatches; ++i) {
            // Get the batch
            auto batch = train.getBatch(i, batchSize);
            auto x = batch.first;
            auto y = batch.second;


            // Zero gradients
            optimizer.zero_grad();

            // Forward pass
            torch::Tensor y_pred = model.forward(x);

            // Compute Loss
            torch::Tensor loss = torch::mse_loss(y_pred, y);
            std::cout << "Epoch: " << epoch << " | Batch Loss: " << loss.item<float>() << std::endl;

            // Backward pass
            loss.backward();

            // Update the parameters
            optimizer.step();
        }
    }

    model.eval();

    torch::Tensor y_pred = model.forward(test.x);
    torch::Tensor loss = torch::mse_loss(y_pred, test.y);
    std::cout << "Test Loss: " << loss.item<float>() << std::endl;
}