#include <cmath>
#pragma once // or use include guards

struct Dataset {
    torch::Tensor x;
    torch::Tensor y; 
    // x and y must have same size at dimension 0


    // Method to get a batch for a given epoch and batch size
    std::pair<torch::Tensor, torch::Tensor> getBatch(int epoch, int batchSize) const {
        // Calculate the starting index of the batch
        int nSamples = x.sizes()[0];
        int startIndex = epoch * batchSize;
        int endIndex =  std::min(nSamples, (startIndex+batchSize));



        // Extract a batch from the dataset
        torch::Tensor batchX = x.index({torch::indexing::Slice(startIndex, startIndex + batchSize)});
        torch::Tensor batchY = y.index({torch::indexing::Slice(startIndex, startIndex + batchSize)});

        return {batchX, batchY};
    }

    Dataset sample(float propDataUsed){
        // slices data by propDataUsed

        int end = round(propDataUsed * (float)x.sizes()[0]);

        x = x.index({torch::indexing::Slice(0,  end)});
        y = y.index({torch::indexing::Slice(0, end)});

        return Dataset({x,y});
    }
};


Dataset prepData(std::string path, float prop);

void printTensor(torch::Tensor tensor);