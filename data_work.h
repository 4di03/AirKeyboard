#include <cmath>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
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


    std::vector<Dataset> slice(int splitInd){

        int len = x.sizes()[0];

        auto xFirst = x.index({torch::indexing::Slice(0,  splitInd)});
        auto yFirst = y.index({torch::indexing::Slice(0,  splitInd)});

        auto xSecond = x.index({torch::indexing::Slice(splitInd ,len)});
        auto ySecond = y.index({torch::indexing::Slice(splitInd, len)});

        std::vector<Dataset> ans;

        ans.push_back(Dataset({xFirst,yFirst}));
        ans.push_back(Dataset({xSecond,ySecond}));


        return ans;

    }

    std::vector<Dataset> sliceProp(float prop){
        int len = x.sizes()[0];

        int splitInd = round(prop*len);

        return slice(splitInd);

    }
    Dataset shuffle() {
        // Get the number of samples
        int numSamples = x.sizes()[0];

        // Generate a random permutation of indices
        torch::Tensor indices = torch::randperm(numSamples, x.options().dtype(torch::kLong));

        // Shuffle the tensors based on the indices
        torch::Tensor shuffledX = torch::index_select(x, 0, indices);
        torch::Tensor shuffledY = torch::index_select(y, 0, indices);

        // Create a new shuffled Dataset
        Dataset shuffledDataset{shuffledX, shuffledY};

        return shuffledDataset;
    }

    void to(torch::Device device){
        x= x.to(device);
        y= y.to(device);
    }



};


Dataset prepData(std::string path, float prop, bool excludeMerged);

void printTensor(const torch::Tensor tensor);
torch::Tensor getKPFromHeatmap(const torch::Tensor& heatmapStack, torch::Device device);
void saveImageToFile(const cv::Mat& image, const std::string& filePath);
cv::Mat tensorToMat(const torch::Tensor& tensor);
cv::Mat drawKeypoints(cv::Mat image, const torch::Tensor kp2d, cv::Scalar color);

bool isSubstringPresent(const std::string mainString, const std::string searchString);

std::tuple<torch::Tensor, torch::Tensor> getNormParams(const torch::Tensor& imageBatch);