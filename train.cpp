#include <iostream>
#include <torch/torch.h>
#include "data_work.h"
#include <cmath>

using namespace std;

class ConvBlock : public torch::nn::Module {
private:
    torch::nn::BatchNorm2d batchNorm1 = nullptr, batchNorm2 = nullptr;
    torch::nn::Conv2d conv1 = nullptr, conv2 = nullptr;
    torch::nn::Functional relu1 = nullptr, relu2 = nullptr;
    torch::nn::Sequential block = nullptr;
    bool debug;
public:
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
        if (debug) {
            cout << "in CONV BLOCK forward" << debug <<  endl;

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
    ConvBlock conv1, conv2, conv3, conv4, conv5, conv6, conv7;
    torch::nn::ConvTranspose2d upsample1 = nullptr;
    torch::nn::ConvTranspose2d upsample2 = nullptr;
    torch::nn::ConvTranspose2d upsample3 =  nullptr;
    torch::nn::Conv2d finalConv = nullptr;
    torch::nn::Functional sigmoid = nullptr;
    torch::nn::Sequential model;

    bool debug;
public:

    ResidualUNet(int nChannels, int initNeurons, bool debug = false) : debug(debug)
    {

        conv1 = ConvBlock(nChannels, initNeurons);
        conv2 = ConvBlock(initNeurons, initNeurons * 2);
        conv3 = ConvBlock(initNeurons * 2, initNeurons * 4);
        conv4 = ConvBlock(initNeurons * 4, initNeurons * 8);
        conv5 = ConvBlock(initNeurons * 12, initNeurons * 4);
        conv6 = ConvBlock(initNeurons * 6, initNeurons * 2);
        conv7 = ConvBlock(initNeurons * 3, initNeurons);

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
            torch::Tensor conv1Out = conv1.forward(x);
            cout << "conv1Out: " << conv1Out.sizes() << endl;


//            torch::Tensor upsampled1 = upsample1->forward(conv1Out);
//            cout << "upsampled1: " << upsampled1.sizes() << endl;

            torch::Tensor conv2Out = conv2.forward(conv1Out);
            cout << "conv2Out: " << conv2Out.sizes() << endl;
//
//            torch::Tensor upsampled2 = upsample2->forward(conv2Out);
//            cout << "upsampled2: " << upsampled2.sizes() << endl;

            torch::Tensor conv3Out = conv3.forward(conv2Out);
            cout << "conv3Out: " << conv3Out.sizes() << endl;

//            torch::Tensor upsampled3 = upsample3->forward(conv3Out);
//            cout << "upsampled3: " << upsampled3.sizes() << endl;

            torch::Tensor conv4Out = conv4.forward(conv3Out);
            cout << "conv4Out: " << conv4Out.sizes() << endl;


            torch::Tensor skipConnected1 = torch::cat({conv3Out,conv4Out},1);  // Concat along columns axis (3rd dimensions) conv4Out + upsampled3;
            cout << "skipConnected1: " << skipConnected1.sizes() << endl;

            torch::Tensor conv5Out = conv5.forward(skipConnected1);


            cout << "conv5Out: " << conv5Out.sizes() << endl;

            torch::Tensor skipConnected2 = torch::cat({conv2Out,conv5Out }, 1);
            cout << "skipConnected2: " << skipConnected2.sizes() << endl;


            torch::Tensor conv6Out = conv6.forward(skipConnected2);
            cout << "conv6Out: " << conv6Out.sizes() << endl;

            torch::Tensor skipConnected3 = torch::cat({conv1Out, conv6Out },1);
            cout << "skipConnected3: " << skipConnected3.sizes() << endl;

            torch::Tensor conv7Out = conv7.forward(skipConnected3);
            cout << "conv7Out: " << conv7Out.sizes() << endl;

            torch::Tensor finalOut = finalConv->forward(conv7Out);
            cout << "finalOut: " << finalOut.sizes() << endl;


            return sigmoid->forward(finalOut);
        } else{

            torch::Tensor conv1Out = conv1.forward(x);
            torch::Tensor conv2Out = conv2.forward(conv1Out);
            torch::Tensor conv3Out = conv3.forward(conv2Out);
            torch::Tensor conv4Out = conv4.forward(conv3Out);

            torch::Tensor skipConnected1 = torch::cat({conv3Out, conv4Out}, 1);

            torch::Tensor conv5Out = conv5.forward(skipConnected1);

            torch::Tensor skipConnected2 = torch::cat({conv2Out, conv5Out}, 1);

            torch::Tensor conv6Out = conv6.forward(skipConnected2);

            torch::Tensor skipConnected3 = torch::cat({conv1Out, conv6Out}, 1);

            torch::Tensor conv7Out = conv7.forward(skipConnected3);

            torch::Tensor finalOut = finalConv->forward(conv7Out);

            return sigmoid->forward(finalOut);


        }
    }


};


void trainModel(Dataset train, Dataset test) {
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

    auto sizes = train.x.sizes();
    int c = sizes[1];
    int nTrainSamples = sizes[0];
    int initNeurons = 16;
    int batchSize = 64;
    ResidualUNet model = ResidualUNet(c, initNeurons);

//    std::vector<torch::Tensor> trainTensors = {train.x,train.y};
//    auto train_loader = torch::data::make_data_loader(
//            torch::data::datasets::TensorDataset(trainTensors),
//            torch::data::DataLoaderOptions().batch_size(64).workers(2)
//    );

    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.01));

    for (size_t epoch = 1; epoch <= 100; ++epoch) {
        // Iterate over batches
        int nBatches = round(nTrainSamples / batchSize);
        for (int i = 0; i < nBatches; ++i) {
            // Get the batch
            auto batch = train.getBatch(i, batchSize);
            auto x = batch.first;
            auto y = batch.second;


            // Zero gradients
            optimizer.zero_grad();


            cout << "x shape: " << x.sizes() << endl;
            // Forward pass

            //Time forward pass

            auto start = std::chrono::high_resolution_clock::now();
            torch::Tensor y_pred = model.forward(x);
            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start;
            std::cout << "Elapsed time for forward pass: " << elapsed.count() << " s\n";


             cout << "y_pred shape: " << y_pred.sizes() << endl;
             cout << "y shape: " << y.sizes() << endl;
            // Compute Loss
            torch::Tensor loss = torch::mse_loss(y_pred, y);
            std::cout << "Batch: " << i << " | Batch Loss: " << loss.item<float>() << std::endl;

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