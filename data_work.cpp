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
#include <xtensor-blas/xlinalg.hpp>
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <json.hpp>

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

    // cout << "Tensor(pos-conversion): " << endl;
    // printTensor(tensor);

    // cout << "Element-wise comparison: " << endl;

    // for (int i = 0; i < rows; i++){
    //     for (int j = 0; j < cols; j++){
    //         cout << vec[i][j] << ", " << tensor[i][j].item<float>() << ", ";
    //       //  cout << (vec[i][j] == tensor[i][j].item<double>()) << ", ";
    //     }
    //     cout << endl;
    // }

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

    cout << "L88: " << getTensorString(kp_xyz_cam) << endl;


    // Normalizing kp_xyz_cam by dividing by the last column
    kp_xyz_cam = kp_xyz_cam / kp_xyz_cam.index({torch::indexing::Slice(), -1}).unsqueeze(1);


    cout << "L90: " << getTensorString(kp_xyz_cam) << endl;

    // Computing kp_uv = kp_xyz_cam * K.T
    torch::Tensor kp_uv = torch::mm(kp_xyz_cam, K);


    cout << "L93: " << getTensorString(kp_uv) << endl;

    // Normalizing kp_uv by dividing by the last column
    kp_uv = kp_uv.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}) / kp_uv.index({torch::indexing::Slice(), -1}).unsqueeze(1);


    cout << "final kp_uv: " << getTensorString(kp_uv) << endl;

    return kp_uv;
}

void prepData(std::string path){
    std::ifstream file(path);
    auto data = xt::load_csv<std::string>(file);

    const auto& column_names = xt::view(data,0, xt::all());

    std::unordered_map<std::string, int> colMap;
    for (int i = 0; i < column_names.size(); i++){
        colMap[column_names[i]] = i;
    }


    for (int i = 1; i < data.shape()[0]; ++i) {
        auto row = xt::view(data, i, xt::all());

        std::string imagePath = row[colMap["image_file"]];
        std::string keypointPath = row[colMap["kp_data_file"]];
        std::string calibPath = row[colMap["calib_file"]];

        cv::Mat image = cv::imread(imagePath);
     
        nlohmann::json kpJson= loadJson(keypointPath);
        nlohmann::json calibJson = loadJson(calibPath);

        auto kpData = kpJson.get<std::vector<std::vector<float>>>(); // careful , converting double to float here

        int cid = extractCameraId(imagePath);
        cout << "imagePath: " << imagePath<<  "Camera id: " << cid << endl;

        //printType(kpData);

        // Convert JSON data to std::vector of std::vector
        auto mVec = calibJson["M"][cid].get<std::vector<std::vector<float>>>();// careful , converting double to float here
        auto kVec = calibJson["K"][cid].get<std::vector<std::vector<float>>>();// careful , converting double to float here
       // Convert std::vector of std::vector to xt::xarray


        // auto mArray =xt::adapt(mVec);
        // auto kArray = xt::adapt(kVec);
        // auto kpDataArray = xt::adapt(kpData);


        auto mArray = vectorToTensor(mVec);
        auto kArray = vectorToTensor(kVec);
        auto kpDataArray = vectorToTensor(kpData);
            
            
        // printType(mArray);
        // printType(kArray); // adapted to be xarrays
//
        cout<< "M: " << endl;
        printTensor(mArray);
        cout<< "K: " << endl;
        printTensor(kArray);

        auto kp2d = get2dKeypoints(kpDataArray, mArray, kArray);

        image = drawKeypoints(image, kp2d);

        cv::imshow("image", image);
        cv::waitKey(0);

    }   

}
