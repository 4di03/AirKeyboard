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
#include <typeinfo>
#include <opencv2/opencv.hpp>
#include <json.hpp>



using namespace std;



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


void prepData(std::string path){
    std::ifstream file(path);
    auto data = xt::load_csv<std::string>(file);

    for (auto& el : data.shape()){
        std::cout << el << ", "; 
    }

    const auto& column_names = xt::view(data,0, xt::all());

    std::unordered_map<std::string, int> colMap;
    for (int i = 0; i < column_names.size(); i++){
        colMap[column_names[i]] = i;
    }

    std::cout << column_names << std::endl;

    for (int i = 1; i < data.shape()[0]; ++i) {
        auto row = xt::view(data, i, xt::all());

        std::string imagePath = row[colMap["image_file"]];
        std::string keypointPath = row[colMap["kp_data_file"]];
        std::string calibPath = row[colMap["calib_file"]];

        cv::Mat image = cv::imread(imagePath);
     
        nlohmann::json kpJson= loadJson(keypointPath);
        nlohmann::json calibJson = loadJson(calibPath);

        auto kpData = kpJson.get<std::vector<std::vector<double>>>();

        int cid = extractCameraId(imagePath);
        cout << "Camera id: " << cid << endl;

        printType(kpData);

        // Convert JSON data to std::vector of std::vector
        std::vector<std::vector<double>> mVec = calibJson["M"][cid].get<std::vector<std::vector<double>>>();
        std::vector<std::vector<double>> kVec = calibJson["K"][cid].get<std::vector<std::vector<double>>>();
       // Convert std::vector of std::vector to xt::xarray
        auto mArray =xt::adapt(mVec);
        auto kArray = xt::adapt(kVec);
        auto kpDataArray = xt::adapt(kpData);
            
        printType(mArray);
        printType(kArray);

        cout<< "M: " << endl;
        for (auto& el : mArray){
            for (auto& el2 : el){
                cout << el2 << ", ";
            }
            cout << endl;
        }
        cout<< "K: " << endl;  
        for (auto& el : kArray){
            for (auto& el2 : el){
                cout << el2 << ", ";
            }
            cout << endl;
        }



        cv::imshow("image", image);
        cv::waitKey(0);

    }   

}
