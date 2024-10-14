#include <torch/torch.h>
#include <torch/script.h>
#include "utils.h"
#include "model_utils.h"

void saveModel(const std::string& modelPath, torch::nn::Module& model){
     try {
        //model = torch::load(model,model_path)
        cout << "SAVING MODEL AT " << modelPath << std::endl;
        // creates the directory for the file if it doesn't exist 
        createDirectory(getDirectoryName(modelPath));


        torch::serialize::OutputArchive output_archive;
        model.save(output_archive);
        output_archive.save_to(modelPath);
        //model = torch::jit::load(model_path);
    } catch (const c10::Error& e) {
        std::cerr << "Error saving the model: " << e.what() << std::endl;
    }
}
// Function to print the device of a PyTorch tensor
void printTensorDevice(const torch::Tensor& tensor) {
    // Get the device of the tensor
    torch::Device device = tensor.device();

    cout << device <<std::endl;
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

// Function to transform a stack of images such that the mean value of each pixel on each channel is 0 and its standard deviation is 1
torch::Tensor standardizeImages(const torch::Tensor x, const torch::Tensor& means, const torch::Tensor& stds){
    // Check if the dimensions match

    auto images = x.clone();

    if (images.size(1) != 3) {
        throw std::invalid_argument("Input images must have 3 channels");
    }

    
  //  cout <<  "L25  " << images.sizes() <<std::endl;
    // Permute the dimensions to bring channels to the front
    images = images.permute({0, 2, 3, 1}).contiguous();

    // // Convert to float and normalize
    // images = images.to(torch::kFloat32) / 255.0;

    // Iterate over channels and apply normalization
    for (int i = 0; i < 3; ++i) {

        auto dev = std::max(1e-8, stds[i].item<double>());
        images.select(3, i).sub_(means[i]).div_(dev);
    }

    // Permute the dimensions back to the original order
    images = images.permute({0, 3, 1, 2}).contiguous();




    // auto np = getNormParams(images);
    // cout << images.sizes() << " MEAN: " << std::get<0>(np) << " STD:"  << std::get<1>(np)<<std::endl;

    return images;//Dataset{images,dset.y};
}


void printComputationGraphAndParams(const torch::jit::script::Module& model) {

    // Print model parameters
    std::cout << "Model Parameters:\n";
    int i = 0;
    for (const auto& param : model.parameters()) {
        i++;
        printTensor(param);

        if (i > 1){
            break;
        }
    }
}
