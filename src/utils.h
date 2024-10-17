#include <torch/torch.h>
#include <vector>
#include <json.hpp>
#pragma once
using namespace std;

template <typename T>
void printVector(const std::vector<T>& vec) {
    for (const auto& el : vec) {
        std::cout << el << ", "; 
    }
    std::cout << std::endl;
}

template <typename T>
void printType(const T& object) {
    std::cout << typeid(object).name() << std::endl;
}



std::string getTensorString(torch::Tensor tensor);
void printTensor(const torch::Tensor tensor);

std::string getModelPath(std::string modelName, std::string fileName = "final_model.pt");
std::string getDirectoryName(const std::string& fileName);
void createDirectory(const std::string dirPath);
template <typename G>
G getOrDefault(const nlohmann::json& jsonObj, const std::string& key, const G& defaultValue) {
    if (jsonObj.contains(key)) {
        // If the key exists, return its value (ensure the type matches the default)
        try {
            return jsonObj.at(key).get<G>();
        } catch (nlohmann::json::type_error& e) {
            std::cerr << "Type error: " << e.what() << std::endl;
            // Return default value if the type doesn't match
            return defaultValue;
        }
    }
    // Return default if key doesn't exist
    return defaultValue;
}
int countNaNs(const torch::Tensor& tensor);
void saveJsonToFile(nlohmann::json jsonObj, std::string jsonSavePath);