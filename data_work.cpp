#include <iostream>
#include <fstream>
#include <torch/torch.h>
#include "data_work.h"
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xcsv.hpp>
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <json.hpp>
#include <cmath>
#include <omp.h>

using namespace std;


std::string getTensorString(torch::Tensor tensor){
    std::ostringstream stream;
    stream << tensor;
    std::string tensor_string = stream.str();

    return tensor_string;
}

void printTensor(torch::Tensor tensor){

    std::string tensor_string = getTensorString(tensor);
    std::cout << tensor_string << std::endl;

}

template <typename T>
void printType(const T& object) {
    std::cout << typeid(object).name() << std::endl;
}

int charToInt(char c){
    return c - '0';
}

void csvToVector(std::string csvPath){

    std::ifstream file(csvPath);

    if (!file.is_open()){
        std::cout << "Error: Could not open file." << std::endl;
        return;
    }

    //load file from path
    std::string line, word;

    while (std::getline(file, line)){
        std::cout << line << std::endl;
    }

}

nlohmann::json loadJson(std::string jsonPath){
    std::ifstream jsonFile(jsonPath);
    nlohmann::json jsonData;
    jsonFile >> jsonData;
    return jsonData;
}

int extractCameraId(std::string imagePath){

    std::string str = imagePath /* your string */;
    std::istringstream ss(str);
    std::string token;
    std::vector<std::string> parts;
    while (std::getline(ss, token, '/')) {
        if (token.find("cam")!= std::string::npos){

            int cid = charToInt(token[token.size()-1]);
            return cid;
        }
    }

    throw std::runtime_error("Could not extract camera id from image path");
}

cv::Mat tensorToMat(torch::Tensor tensor){
    /**
     * @brief convert a tensor to a cv::Mat
     *
     */
    if (!tensor.is_contiguous() || !tensor.is_cpu()) {
        // You may need to make the tensor contiguous and move it to CPU if necessary
        tensor = tensor.contiguous().to(torch::kCPU);
    }

    // Get the tensor dimensions
    int height = tensor.size(0);
    int width = tensor.size(1);
    int channels = tensor.size(2);

    // Initialize a cv::Mat with the appropriate data type
    cv::Mat mat(height, width, channels == 1 ? CV_8UC1 : CV_8UC3);

    // Convert the tensor data type to INT and copy data to the cv::Mat
    tensor = tensor.to(torch::kByte);
    std::memcpy(mat.data, tensor.data_ptr(), sizeof(uint8_t) * tensor.numel());

    return mat.clone(); // Ensure a deep copy to avoid memory issues
}
torch::Tensor matToTensor(const cv::Mat& image){

    // Check if the image is empty
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty.");
    }
    int nChannels = image.channels();

    // Convert OpenCV Mat to PyTorch Tensor
    torch::Tensor tensor_image = torch::from_blob(image.data, {1, image.rows, image.cols, nChannels}, at::kByte).clone();

    // Reshape the tensor to (C, H, W) format
    tensor_image = tensor_image.view({ image.rows, image.cols,nChannels});


    tensor_image = tensor_image.to(at::kFloat);
//    // Normalize the pixel values to the range [0, 1]
//    tensor_image = tensor_image.to(torch::kFloat32) / 255.0;

//    // Transpose the tensor to (H, W, C) format
//    tensor_image = tensor_image.permute({1, 2, 0});


    return tensor_image; // Ensure a deep copy to avoid memory issues
}


torch::Tensor vectorToTensor(std::vector<std::vector<float>> vec){
    /**
     * @brief convert a vector of arbitrary rank to an xarray
     * 
     */

    // cout << "Vector(pre-conversion): " << endl;
    // for (auto& el : vec){
    //     for (auto& el2 : el){
    //         cout << el2 << ", ";
    //     }
    //     cout << endl;
    // }

    std::vector<float> flattened;

    for (auto el : vec){
        for (float el2 : el){
            flattened.push_back(el2);
        }
    }
    int rows = vec.size();
    int cols = vec[0].size();



    torch::Tensor tensor = torch::from_blob(flattened.data(), {rows, cols}).clone();

    return tensor; // remove dependency on flattened vector
}


cv::Mat drawKeypoints(cv::Mat image, torch::Tensor kp2d){
    /**
     * @brief draw keypoints on image
     *
     */

    // cout << "kp2d: " << endl;
    // printTensor(kp2d);

    int numKeypoints = kp2d.size(0);

    for (int i = 0; i < numKeypoints; i++){
        int x = kp2d[i][0].item<int>();
        int y = kp2d[i][1].item<int>();
        cv::circle(image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
    }

    return image;
}



torch::Tensor get2dKeypoints(torch::Tensor kp3dArray, torch::Tensor mArray, torch::Tensor kArray){
    /**
     * @brief project 3d keypoints to 2d using camera matrix and intrinsic matrix
     * 
     */

    // Extracting relevant submatrices from M_w2cam and K matrices
    torch::Tensor M_w2cam = mArray.index({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)});
    torch::Tensor K = kArray.t();

    // Computing kp_xyz_cam = kp_xyz * M_w2cam[:3, :3].T + M_w2cam[:3, 3][None]

    auto mm = torch::mm(kp3dArray, M_w2cam.t());
    auto mCamSlice =  mArray.index({torch::indexing::Slice(0,3),3}).unsqueeze(0);
    torch::Tensor kp_xyz_cam = mm +mCamSlice;



    // Normalizing kp_xyz_cam by dividing by the last column
    kp_xyz_cam = kp_xyz_cam / kp_xyz_cam.index({torch::indexing::Slice(), -1}).unsqueeze(1);



    // Computing kp_uv = kp_xyz_cam * K.T
    torch::Tensor kp_uv = torch::mm(kp_xyz_cam, K);



    // Normalizing kp_uv by dividing by the last column
    kp_uv = kp_uv.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}) / kp_uv.index({torch::indexing::Slice(), -1}).unsqueeze(1);



    return kp_uv;
}

torch::Tensor resizeKeypoints(torch::Tensor kp2d ,const std::vector<int> origSize, const std::vector<int> newSize){

    for (int i = 0; i < kp2d.size(0); i++){
        kp2d.index({i, 0}) = kp2d.index({i, 0}) * newSize[0] / origSize[0];
        kp2d.index({i, 1}) = kp2d.index({i, 1}) * newSize[1] / origSize[1];
    }

    return kp2d;

}



torch::Tensor getJointHeatmaps(const torch::Tensor& kp2d, const std::vector<int> imgSize){
    int w = imgSize[0];
    int h = imgSize[1];

    vector<torch::Tensor> heatmaps;
    for (int i = 0 ;i < kp2d.size(0); i++){
        int kpX= kp2d.index({i, 0}).item<int>();
        int kpY = kp2d.index({i, 1}).item<int>();
        cv::Mat heatmap = cv::Mat::zeros(w,h,CV_32FC1);
        heatmap.at<float>(kpY, kpX) = 1.0;


        cv::GaussianBlur(heatmap, heatmap, cv::Size(5,5), 0);
        float maxPixel = *std::max_element(heatmap.begin<float>(),heatmap.end<float>());
        heatmap = heatmap / maxPixel; // normalization so that we can sigmoid it with model


        torch::Tensor heatmapTensor = matToTensor(heatmap);
        heatmaps.push_back(heatmapTensor);
    }

    torch::Tensor heatmapsTensor = torch::stack(heatmaps);
    return heatmapsTensor;

}

vector<torch::Tensor> removeNullVectors(vector<torch::Tensor> v){
    vector<torch::Tensor> ret;

    for (auto t : v){
        if (t.sizes().size() > 1){
            ret.push_back(t);
        }
    }

    return ret;

}

Dataset prepData(std::string path, float prop = 1.0){
    std::ifstream file(path);
    auto data = xt::load_csv<std::string>(file);

    int nRows = round(data.shape()[0] * prop);

    data = xt::view(data, xt::range(0, nRows), xt::all()); // take first nRows

    const auto& column_names = xt::view(data,0, xt::all());

    std::unordered_map<std::string, int> colMap;
    for (int i = 0; i < column_names.size(); i++){
        colMap[column_names[i]] = i;
    }

    cout << "Processing "<< data.shape()[0] << " Images" << endl;

    int nDataPoints = data.shape()[0];
    vector<torch::Tensor> xData(nDataPoints);
    vector<torch::Tensor> yData(nDataPoints);

    #pragma omp parallel for
    for (int i = 1; i < nDataPoints; ++i) {
        auto row = xt::view(data, i, xt::all());

        std::string imagePath = row[colMap["image_file"]];
        std::string keypointPath = row[colMap["kp_data_file"]];
        std::string calibPath = row[colMap["calib_file"]];

        if ( i % 100 == 0) {
            cout << "Processing Image " << i << endl;
        }
        cv::Mat image;
        try {
            image = cv::imread(imagePath);

            if (image.empty()){
                throw std::exception();
            }
        } catch (std::exception& e){
            cout << "Could not read image: " << imagePath << endl;
            continue;
        }

        nlohmann::json kpJson= loadJson(keypointPath);
        nlohmann::json calibJson = loadJson(calibPath);

        auto kpData = kpJson.get<std::vector<std::vector<float>>>(); // careful , converting double to float here

        int cid = extractCameraId(imagePath);


        // Convert JSON data to std::vector of std::vector
        auto mVec = calibJson["M"][cid].get<std::vector<std::vector<float>>>();// careful , converting double to float here
        auto kVec = calibJson["K"][cid].get<std::vector<std::vector<float>>>();// careful , converting double to float here
       // Convert std::vector of std::vector to xt::xarray

        auto mArray = vectorToTensor(mVec);
        auto kArray = vectorToTensor(kVec);
        auto kpDataArray = vectorToTensor(kpData);

        auto kp2d = get2dKeypoints(kpDataArray, mArray, kArray);



        cv::Size s = image.size();

        cv::Mat shrunkImage;
        cv::resize(image,shrunkImage,cv::Size(128,128)); // condense for model

        torch::Tensor shrunk_kp2d = resizeKeypoints(kp2d, {s.width, s.height}, {128,128});

        image = drawKeypoints(shrunkImage, shrunk_kp2d);


        torch::Tensor imageTensor = matToTensor(shrunkImage);

        torch::Tensor jointHeatmaps  = getJointHeatmaps(shrunk_kp2d, {128,128}); // gets 21x128x128 tensor where each of the 2d tensors is a heatmap for each keypoint
        
        // Inside the loop, assign to the vectors in a thread-safe manner
        #pragma omp critical
        {
            xData[i - 1] = imageTensor;
            yData[i - 1] = jointHeatmaps;
        }
        
        //xData.push_back(imageTensor);
        //yData.push_back(jointHeatmaps);

    }

    xData = removeNullVectors(xData);
    yData = removeNullVectors(yData);
    torch::Tensor xTensor = torch::stack(xData);
    torch::Tensor yTensor = torch::stack(yData);


    Dataset d;
    d.x = xTensor;
    d.y = yTensor;
    return d;

}
