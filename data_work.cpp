#include <iostream>
#include <fstream>
#include <torch/torch.h>
#include "data_work.h"
#include <vector>
#include <xtensor/xarray.hpp>
<<<<<<< HEAD
#include <xtensor/xadapt.hpp>
=======
>>>>>>> 9faf08865b02c831689cb0fbc1434c782d8b2966
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xcsv.hpp>
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <json.hpp>
#include <cmath>
<<<<<<< HEAD
=======
#include <opencv2/imgcodecs.hpp>
#include <omp.h>

>>>>>>> 9faf08865b02c831689cb0fbc1434c782d8b2966

using namespace std;


std::string getTensorString(torch::Tensor tensor){
    std::ostringstream stream;
    stream << tensor;
    std::string tensor_string = stream.str();

    return tensor_string;
}

<<<<<<< HEAD
void printTensor(torch::Tensor tensor){
=======
void printTensor(const torch::Tensor tensor){
>>>>>>> 9faf08865b02c831689cb0fbc1434c782d8b2966

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

<<<<<<< HEAD
cv::Mat tensorToMat(torch::Tensor tensor){
    /**
     * @brief convert a tensor to a cv::Mat
     *
     */
=======

std::tuple<torch::Tensor, torch::Tensor> getNormParams(const torch::Tensor& imageBatch) {

    //cout << "L90 get norma params tensor input max val: " << torch::max(imageBatch).item<float>() << endl;

    torch::Device device = imageBatch.device();

    torch::Tensor mean = torch::zeros({3}).to(device);
    torch::Tensor std = torch::zeros({3}).to(device);
    double nb_samples = 0.0;

    auto data = imageBatch;

    auto batch_samples = data.size(0);
    data = data.view({batch_samples, data.size(1), -1});
    mean += data.mean({2}).sum({0});
    std += data.std({2}).sum({0});
    nb_samples += batch_samples;

    mean /= nb_samples;
    std /= nb_samples;

    return std::make_tuple(mean, std);
}


cv::Mat tensorToMat(const torch::Tensor& trueTensor){
    /**
     * @brief convert a tensor to a cv::Mat
     * tensor should have dimension (C, H, W)
     * CREATES FLOAT IF SINGLE CHANNEL IMAGE, ELSE INT
     */
    auto sizes = trueTensor.sizes();
    int c = sizes[0];
    int h = sizes[1];
    int w = sizes[2];

    //cout <<"L96 " <<  c  << " "<< h  << " " << w << endl;
    auto tensor = trueTensor.view({h,w,c});// transform it to shape required for cv::Mat

>>>>>>> 9faf08865b02c831689cb0fbc1434c782d8b2966
    if (!tensor.is_contiguous() || !tensor.is_cpu()) {
        // You may need to make the tensor contiguous and move it to CPU if necessary
        tensor = tensor.contiguous().to(torch::kCPU);
    }

    // Get the tensor dimensions
    int height = tensor.size(0);
    int width = tensor.size(1);
    int channels = tensor.size(2);

<<<<<<< HEAD
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
=======
    // // Initialize a cv::Mat with the appropriate data type
    // cv::Mat mat(height, width, channels == 1 ? CV_8UC1 : CV_8UC3);
    auto torchDTYPE =  tensor.dtype();//channels == 1 ? torch::kFloat : torch::kByte;

    auto cvDTYPE = ( torchDTYPE == torch::kFloat ? (channels == 1 ? CV_32FC1: CV_32FC3) : (channels == 1 ? CV_8UC1 : CV_8UC3));//channels == 1 ? CV_32FC1: CV_8UC3;
    // Convert the tensor data type to INT and copy data to the cv::Mat
    tensor = tensor.to(torchDTYPE);

    tensor = tensor.view({h,w,c});

    //cout << "Input Tensor size: " << tensor.sizes() << endl;

    // Create a cv::Mat directly from tensor data
    cv::Mat mat(height, width, cvDTYPE, tensor.data_ptr());

    //cout << "Converted Mat size : " << mat.size() << endl;
    return mat.clone(); // Ensure a deep copy to avoid memory issues
}

template <typename T>
T findMaxValue(const cv::Mat& image) {
    // Check if the image is empty
    if (image.empty()) {
        std::cerr << "Input image is empty!" << std::endl;
        return 0.0;  // Return a default value (you may choose a different approach)
    }

    // Find the maximum value in the image
    T maxVal;
    cv::minMaxLoc(image, nullptr, &maxVal);

    return maxVal;
}

torch::Tensor matToTensor(const cv::Mat& image){
    // returns tenosr in (C,H,W format)
    // Check if the image is empty  
    //  reads floats for single channel image (0-1), bytes(ints) for BGR images
    if (image.empty()) {
>>>>>>> 9faf08865b02c831689cb0fbc1434c782d8b2966
        throw std::invalid_argument("Input image is empty.");
    }
    int nChannels = image.channels();

<<<<<<< HEAD
    // Convert OpenCV Mat to PyTorch Tensor
    torch::Tensor tensor_image = torch::from_blob(image.data, {1, image.rows, image.cols, nChannels}, at::kByte).clone();

    // Reshape the tensor to (C, H, W) format
    tensor_image = tensor_image.view({ image.rows, image.cols,nChannels});


    tensor_image = tensor_image.to(at::kFloat);
//    // Normalize the pixel values to the range [0, 1]
//    tensor_image = tensor_image.to(torch::kFloat32) / 255.0;

//    // Transpose the tensor to (H, W, C) format
//    tensor_image = tensor_image.permute({1, 2, 0});

=======
    auto imgType = image.depth();  // Get the depth of the image (CV_8U, CV_32F, etc.)
    c10::ScalarType dtype;
    if (imgType == CV_32F || imgType == CV_64F) {
        // Image is float, choose at::kFloat for PyTorch tensor
        dtype = at::kFloat;
    } else {
        // Image is integer, choose at::kByte for single-channel or at::kRGB for BGR images
        dtype = at::kByte;//(nChannels == 1) ? at::kByte;
    }
   // cout << " L166 : " << findMaxValue<double>(image) << " DTYPE: " << dtype  << " " << endl;

    // Convert OpenCV Mat to PyTorch Tensor
    //cout << "L166.2 " <<  image.data.size() << endl;
    torch::Tensor tensor_image = torch::from_blob(image.data, {image.rows, image.cols, nChannels}, dtype).clone();
   // cout << "L166.5 : " <<  torch::max(tensor_image).item<float>();

    // Reshape the tensor to (C, H, W) format
    tensor_image = tensor_image.view({ nChannels,image.rows, image.cols});
   // cout << " L167 : " <<  torch::max(tensor_image).item<float>() << endl;;


    tensor_image = tensor_image.to(at::kFloat);
>>>>>>> 9faf08865b02c831689cb0fbc1434c782d8b2966

    return tensor_image; // Ensure a deep copy to avoid memory issues
}


torch::Tensor vectorToTensor(std::vector<std::vector<float>> vec){
    /**
     * @brief convert a vector of arbitrary rank to an xarray
     * 
     */

<<<<<<< HEAD
    // cout << "Vector(pre-conversion): " << endl;
    // for (auto& el : vec){
    //     for (auto& el2 : el){
    //         cout << el2 << ", ";
    //     }
    //     cout << endl;
    // }
=======
>>>>>>> 9faf08865b02c831689cb0fbc1434c782d8b2966

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


<<<<<<< HEAD
cv::Mat drawKeypoints(cv::Mat image, torch::Tensor kp2d){
=======
void saveImageToFile(const cv::Mat& image, const std::string& filePath) {
    // image is a noramlzied image
    // Check if the input image is empty

    if (image.empty()) {
        std::cerr << "Error: Input image is empty." << std::endl;
        return;
    }

    // Save the image to the specified file
    //cout << "Saving image to " << filePath << endl;

    // cv::Mat img2;
    // image.convertTo(img2, CV_8U, 255, 0);

    cv::imwrite(filePath, image);
    return;
}

cv::Mat drawKeypoints(cv::Mat image, const torch::Tensor kp2d, cv::Scalar color = cv::Scalar(0, 0, 255)){
>>>>>>> 9faf08865b02c831689cb0fbc1434c782d8b2966
    /**
     * @brief draw keypoints on image
     *
     */

    // cout << "kp2d: " << endl;
    // printTensor(kp2d);

    int numKeypoints = kp2d.size(0);

    for (int i = 0; i < numKeypoints; i++){
<<<<<<< HEAD
        int x = kp2d[i][0].item<int>();
        int y = kp2d[i][1].item<int>();
        cv::circle(image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
=======
        int x = round(kp2d[i][0].item<float>());
        int y = round(kp2d[i][1].item<float>());
        cv::circle(image, cv::Point(x, y), 2, color  , -1);
>>>>>>> 9faf08865b02c831689cb0fbc1434c782d8b2966
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


<<<<<<< HEAD

=======
>>>>>>> 9faf08865b02c831689cb0fbc1434c782d8b2966
torch::Tensor getJointHeatmaps(const torch::Tensor& kp2d, const std::vector<int> imgSize){
    int w = imgSize[0];
    int h = imgSize[1];

    vector<torch::Tensor> heatmaps;
    for (int i = 0 ;i < kp2d.size(0); i++){
        int kpX= kp2d.index({i, 0}).item<int>();
        int kpY = kp2d.index({i, 1}).item<int>();
        cv::Mat heatmap = cv::Mat::zeros(w,h,CV_32FC1);
<<<<<<< HEAD
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

Dataset prepData(std::string path, float prop = 1.0){
=======


        heatmap.at<float>(kpY, kpX) = 1.0;

        cv::GaussianBlur(heatmap, heatmap, cv::Size(51,51), 0);

        float maxPixel = *std::max_element(heatmap.begin<float>(),heatmap.end<float>());

        heatmap = heatmap / maxPixel; // normalization so that we can sigmoid it with model


     
        torch::Tensor heatmapTensor = matToTensor(heatmap);

        heatmaps.push_back(heatmapTensor);

    }

    torch::Tensor heatmapsTensor = torch::stack(heatmaps);


    auto ret = heatmapsTensor.view({kp2d.sizes()[0],h,w});

    return ret;

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

double findMaxValue(const cv::Mat& matrix) {
    double maxVal;
    cv::minMaxLoc(matrix, nullptr, &maxVal);
    return maxVal;
}

bool isSubstringPresent(const std::string mainString, const std::string searchString) {
    size_t found = mainString.find(searchString);
    
    return found != std::string::npos;
}

cv::Mat readImage(std::string imagePath){
    auto img = cv::imread(imagePath);

    cv::imwrite("L413_test.jpg", img);

    cv::Mat img2;
    img.convertTo(img2, CV_32F, 1.0 / 255, 0);

    // cv::Mat img3;
    // img2.convertTo(img3, CV_8U, 255, 0);
    // // Save the normalized and converted image
    cv::imwrite("L414_test_img2.jpg", img2);

    
   // cv::cvtColor(bgrImage, bgrImage, cv::COLOR_BGR2RGB);
    return img2;
}

Dataset prepData(std::string path, float prop = 1.0,bool excludeMerged = false ){
    // exclude merged is whether or not to exclude bg shfited images from trianing 
>>>>>>> 9faf08865b02c831689cb0fbc1434c782d8b2966
    std::ifstream file(path);
    auto data = xt::load_csv<std::string>(file);

    int nRows = round(data.shape()[0] * prop);

    data = xt::view(data, xt::range(0, nRows), xt::all()); // take first nRows

    const auto& column_names = xt::view(data,0, xt::all());

    std::unordered_map<std::string, int> colMap;
    for (int i = 0; i < column_names.size(); i++){
        colMap[column_names[i]] = i;
    }

<<<<<<< HEAD
    vector<torch::Tensor> xData;
    vector<torch::Tensor> yData;

    cout << "Processing "<< data.shape()[0] << " Images" << endl;

    for (int i = 1; i < data.shape()[0]; ++i) {
        auto row = xt::view(data, i, xt::all());

=======
    cout << "Processing "<< data.shape()[0] << " Images" << endl;

    int nDataPoints = data.shape()[0];
    vector<torch::Tensor> xData(nDataPoints);
    vector<torch::Tensor> yData(nDataPoints);

    cout << "N = " << nDataPoints << endl;
    bool printit = false;
    //#pragma omp parallel for
    for (int i = 1; i < nDataPoints; ++i) {
        auto row = xt::view(data, i, xt::all());

    

>>>>>>> 9faf08865b02c831689cb0fbc1434c782d8b2966
        std::string imagePath = row[colMap["image_file"]];
        std::string keypointPath = row[colMap["kp_data_file"]];
        std::string calibPath = row[colMap["calib_file"]];

<<<<<<< HEAD
=======
        
        std::string rbm= "rgb_merged";
        if (excludeMerged && isSubstringPresent(imagePath, rbm)){
            // skip over rgb_merged data
            //cout << "skipping " << imagePath << endl;
            continue;
        }

>>>>>>> 9faf08865b02c831689cb0fbc1434c782d8b2966
        if ( i % 100 == 0) {
            cout << "Processing Image " << i << endl;
        }
        cv::Mat image;
        try {
<<<<<<< HEAD
            image = cv::imread(imagePath);
=======
            image = readImage(imagePath);
>>>>>>> 9faf08865b02c831689cb0fbc1434c782d8b2966

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
<<<<<<< HEAD

        image = drawKeypoints(shrunkImage, shrunk_kp2d);


        torch::Tensor imageTensor = matToTensor(shrunkImage);

        xData.push_back(imageTensor);
        torch::Tensor jointHeatmaps  = getJointHeatmaps(shrunk_kp2d, {128,128}); // gets 21x128x128 tensor where each of the 2d tensors is a heatmap for each keypoint
        yData.push_back(jointHeatmaps);

    }

    torch::Tensor xTensor = torch::stack(xData);
    torch::Tensor yTensor = torch::stack(yData);

=======
        

        cv::imwrite("L504_test_normalized.jpg", shrunkImage);

        cv::imwrite("L504_test.jpg", shrunkImage*255);


       // cout << "L544: pre matToTensor" << findMaxValue<double>(shrunkImage) << endl;

        torch::Tensor imageTensor = matToTensor(shrunkImage);
        //cout << "L545: mid data-prep: " << torch::max(imageTensor).item<float>() << endl;


        //cout << " JOINT HEATMAP START: " << endl;
        torch::Tensor jointHeatmaps  = getJointHeatmaps(shrunk_kp2d, {128,128}); // gets 21x128x128 tensor where each of the 2d tensors is a heatmap for each keypoint

       //cout << " JOINT HEATMAP END: " << endl;


        xData[i - 1] = imageTensor;
        yData[i - 1] = jointHeatmaps;


    }

    xData = removeNullVectors(xData);
    yData = removeNullVectors(yData);
    torch::Tensor xTensor = torch::stack(xData);
    torch::Tensor yTensor = torch::stack(yData);

    cout << "L568: post data-prep: " << torch::max(xTensor).item<float>() << endl;

    std::string tmpFilePath = "/scratch/palle.a/AirKeyboard/data/tmp/pp_rand_hm.jpg";
    // not an issue with jointToHeatMaps , sum remains the same
>>>>>>> 9faf08865b02c831689cb0fbc1434c782d8b2966

    Dataset d;
    d.x = xTensor;
    d.y = yTensor;
    return d;

}
<<<<<<< HEAD
=======


torch::Tensor getKPFromHeatmap(const torch::Tensor& heatmaps, torch::Device device){
    // heatmapStack has shape (21, 128,128) representing all 21 heatmaps
    // will return tensor of shape (21, 2) representing (x,y) point fo reach heatmap
    //cout << "EXTRACTING KP" << endl;
    auto heatmapStack = heatmaps.view({21,128,128}).clone();//.clone();
    //(heatmapStack[0]);

    //cout << "TOTAL SUM: " << heatmapStack.sum({0,1,2}) << endl; // expect 21

    auto heatmapSums = heatmapStack.sum({1,2});



    auto hExpanded = heatmapSums.view({heatmapSums.sizes()[0],1,1});
    // cout << "Heatmap sums" << endl;

    // printTensor(hExpanded);


    heatmapStack /= hExpanded;// standardizing heatmap so all values sum to 1
    
   // printTensor(heatmapStack);


    auto rowSums = heatmapStack.sum({1}); // of shape (21, 128)
    auto colSums = heatmapStack.sum({2}); // of shape (21, 128)

 
    auto range = torch::arange(128).to(device).to(torch::kFloat);

    auto x = torch::mm(rowSums, range.view({128,1}));
    auto y = torch::mm(colSums, range.view({128,1}));

    //printTensor(x);

   // printTensor(y);
    return torch::cat(std::vector<torch::Tensor>({x,y}),1);

}



>>>>>>> 9faf08865b02c831689cb0fbc1434c782d8b2966
