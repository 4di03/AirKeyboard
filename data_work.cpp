#include <iostream>
#include <fstream>
#include <torch/torch.h>
#include "data_work.h"
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xcsv.hpp>
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <json.hpp>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <omp.h>


using namespace std;


std::string getTensorString(torch::Tensor tensor){
    std::ostringstream stream;
    stream << tensor;
    std::string tensor_string = stream.str();

    return tensor_string;
}

void printTensor(const torch::Tensor tensor){

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

    if (!tensor.is_contiguous() || !tensor.is_cpu()) {
        // You may need to make the tensor contiguous and move it to CPU if necessary
        tensor = tensor.contiguous().to(torch::kCPU);
    }

    // Get the tensor dimensions
    int height = tensor.size(0);
    int width = tensor.size(1);
    int channels = tensor.size(2);

    // // Initialize a cv::Mat with the appropriate data type
    // cv::Mat mat(height, width, channels == 1 ? CV_8UC1 : CV_8UC3);


    auto cvDTYPE = channels == 1 ? CV_32FC1: CV_8UC3;
    auto torchDTYPE = channels == 1 ? torch::kFloat : torch::kByte;
    // Convert the tensor data type to INT and copy data to the cv::Mat
    tensor = tensor.to(torchDTYPE);

    tensor = tensor.view({h,w,c});

    //cout << "Input Tensor size: " << tensor.sizes() << endl;

    // Create a cv::Mat directly from tensor data
    cv::Mat mat(height, width, cvDTYPE, tensor.data_ptr());

    //cout << "Converted Mat size : " << mat.size() << endl;
    return mat.clone(); // Ensure a deep copy to avoid memory issues
}


// cv::Mat tensorToMat(torch::Tensor tensor){
//     /**
//      * @brief convert a tensor to a cv::Mat
//      * tensor has dimension (C, H, W)
//      */
//     auto sizes = tensor.sizes();
//     int c = sizes[0];
//     int h = sizes[1];
//     int w = sizes[2];
//     tensor = tensor.view({h,w,c});// transform it to shape required for cv::Mat

//     if (!tensor.is_contiguous() || !tensor.is_cpu()) {
//         // You may need to make the tensor contiguous and move it to CPU if necessary
//         tensor = tensor.contiguous().to(torch::kCPU);
//     }

//     // Get the tensor dimensions
//     int height = tensor.size(0);
//     int width = tensor.size(1);
//     int channels = tensor.size(2);

//     // Initialize a cv::Mat with the appropriate data type
//     cv::Mat mat(height, width, channels == 1 ? CV_8UC1 : CV_8UC3);

//     // Convert the tensor data type to INT and copy data to the cv::Mat
//     tensor = tensor.to(torch::kByte);

//     std::memcpy(mat.data, tensor.data_ptr(), sizeof(uint8_t) * tensor.numel());
//     return mat.clone(); // Ensure a deep copy to avoid memory issues
// }



torch::Tensor matToTensor(const cv::Mat& image){
    // returns tenosr in (C,H,W format)
    // Check if the image is empty  
    //  reads floats for single channel image (0-1), bytes(ints) for BGR images
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty.");
    }
    int nChannels = image.channels();

    auto dtype = (nChannels == 1) ? at::kFloat: at::kByte ; //  reads floats for single channel image (0-1), bytes(ints) for BGR images

    // Convert OpenCV Mat to PyTorch Tensor
    torch::Tensor tensor_image = torch::from_blob(image.data, {1, image.rows, image.cols, nChannels}, dtype).clone();

    // Reshape the tensor to (C, H, W) format
    tensor_image = tensor_image.view({ nChannels,image.rows, image.cols});


    tensor_image = tensor_image.to(at::kFloat);
//    // Normalize the pixel values to the range [0, 1]
//    tensor_image = tensor_image.to(torch::kFloat32) / 255.0;


    return tensor_image; // Ensure a deep copy to avoid memory issues
}


torch::Tensor vectorToTensor(std::vector<std::vector<float>> vec){
    /**
     * @brief convert a vector of arbitrary rank to an xarray
     * 
     */


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


void saveImageToFile(const cv::Mat& image, const std::string& filePath) {
    // Check if the input image is empty
    if (image.empty()) {
        std::cerr << "Error: Input image is empty." << std::endl;
        return;
    }

    // Save the image to the specified file
    //cout << "Saving image to " << filePath << endl;
    cv::imwrite(filePath, image);
    return;
}

cv::Mat drawKeypoints(cv::Mat image, const torch::Tensor kp2d, cv::Scalar color = cv::Scalar(0, 0, 255)){
    /**
     * @brief draw keypoints on image
     *
     */

    // cout << "kp2d: " << endl;
    // printTensor(kp2d);

    int numKeypoints = kp2d.size(0);

    for (int i = 0; i < numKeypoints; i++){
        int x = round(kp2d[i][0].item<float>());
        int y = round(kp2d[i][1].item<float>());
        cv::circle(image, cv::Point(x, y), 2, color  , -1);
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

        //cout << "X: " << kpX << "Y: " << kpY << endl;


    
       // cout << "Previous VALUE " <<  heatmap.at<float>(kpY,kpX) << endl;


        heatmap.at<float>(kpY, kpX) = 1.0;
        // heatmap.at<float>(kpY+1, kpX+1) = 1.0;
        // heatmap.at<float>(kpY-1, kpX+1) = 1.0;
        // heatmap.at<float>(kpY-1, kpX-1) = 1.0;
        // heatmap.at<float>(kpY, kpX+1) = 1.0;
        // heatmap.at<float>(kpY, kpX-1) = 1.0;
        // heatmap.at<float>(kpY+1, kpX) = 1.0;
        // heatmap.at<float>(kpY-1, kpX) = 1.0;


       // cout << "UPDATED VALUE " <<  heatmap.at<float>(kpY,kpX) << endl;

        // cv::Mat intMat;
        // heatmap.convertTo(intMat, CV_8U);
        // Scale the values to the range [0, 255]
        // heatmap *= 255; // for the sake of visualization
        //saveImageToFile(heatmap, "/scratch/palle.a/AirKeyboard/data/tmp/pre_blur_tmp_heatmap_"+std::to_string(i)+".jpg");

        cv::GaussianBlur(heatmap, heatmap, cv::Size(5,5), 0);

        float maxPixel = *std::max_element(heatmap.begin<float>(),heatmap.end<float>());
        // if (i == 0) {

        // cout <<"L340 maxPixel:  " << maxPixel << endl;

        // cout <<"L341 single heatmap sum(pre div):  " << cv::sum(heatmap)[0] << endl;

        // }
        heatmap = heatmap / maxPixel; // normalization so that we can sigmoid it with model


        //saveImageToFile(heatmap*255, "/scratch/palle.a/AirKeyboard/data/tmp/tmp_heatmap_pre_"+std::to_string(i)+".jpg");

     
        torch::Tensor heatmapTensor = matToTensor(heatmap);

        // if (i == 0) {

        // // cout << "L341.5" << heatmapTensor.sizes() << endl;
        // // cout <<"L342 single heatmap sum(pre matToTensor):  " << heatmapTensor.sum(std::vector<int64_t>({0,1,2})).item<float>() << endl;


        // // cout << "L346 " << heatmapTensor.sizes() << endl;
        // // cout <<"single heatmap sum  " << heatmapTensor.sum(std::vector<int64_t>({0,1,2})).item<float>() << endl;


        // // cout << "L347 " << heatmapTensor.sizes() << endl;
        // // cout <<"sum per heatmap: " << endl;

        // // printTensor(heatmapTensor.sum({1,2}));

        // // auto mat = tensorToMat(heatmapTensor);
        // // saveImageToFile(mat*255, "/scratch/palle.a/AirKeyboard/data/tmp/L347_tmp_heatmap_post_"+std::to_string(i)+".jpg");
        // }
        heatmaps.push_back(heatmapTensor);

    }

    torch::Tensor heatmapsTensor = torch::stack(heatmaps);


    auto ret = heatmapsTensor.view({kp2d.sizes()[0],h,w});
    // cout << "L362.5 " << ret[0].sizes() << endl;
    // auto tgtTensor = ret[0].view({1,128,128});
    // cout <<"single heatmap ( pre matToTensor, l363)  sum  " <<tgtTensor.sum({0,1,2}).item<float>() << endl;


    // auto mat = tensorToMat(tgtTensor);

    // cout <<"single heatmap ( post matToTensor, l364)  sum  " << matToTensor(mat).sum({0,1,2}).item<float>() << endl;

    // saveImageToFile(mat*255, "/scratch/palle.a/AirKeyboard/data/tmp/L362_tmp_heatmap_post.jpg");

    //cout << heFatmapsTensor.sizes() << endl;

   // cout << "TOTAL SUM (init): " << heatmapsTensor.sum({0,1,2,3}) << endl; // expect 21

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

bool isSubstringPresent(const std::string& mainString, const std::string& searchString) {
    size_t found = mainString.find(searchString);
    
    return found != std::string::npos;
}

Dataset prepData(std::string path, float prop = 1.0,bool excludeMerged = false ){
    // exclude merged is whether or not to exclude bg shfited images from trianing 
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

    cout << "N = " << nDataPoints << endl;
    #pragma omp parallel for
    for (int i = 1; i < nDataPoints; ++i) {
        auto row = xt::view(data, i, xt::all());

    

        std::string imagePath = row[colMap["image_file"]];
        std::string keypointPath = row[colMap["kp_data_file"]];
        std::string calibPath = row[colMap["calib_file"]];

        
        if (excludeMerged && isSubstringPresent(imagePath, "rgb_merged")){
            // skip over rgb_merged data
            continue;
        }

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

        // auto drawImage = drawKeypoints(shrunkImage, shrunk_kp2d);

        // saveImageToFile(drawImage, "/scratch/palle.a/AirKeyboard/data/tmp/pure_" + std::to_string(i) +".jpg");

        // cout << "L416 testing bidrectional conversion of mat to tensor " << drawImage.size() << endl;

        // saveImageToFile(tensorToMat(matToTensor(drawImage)), "/scratch/palle.a/AirKeyboard/data/tmp/conv_tmp_" + std::to_string(i) +".jpg");

        torch::Tensor imageTensor = matToTensor(shrunkImage);

        torch::Tensor jointHeatmaps  = getJointHeatmaps(shrunk_kp2d, {128,128}); // gets 21x128x128 tensor where each of the 2d tensors is a heatmap for each keypoint
        
        // cout << "L498" << jointHeatmaps[0].sizes() << endl;
        // auto tgt = jointHeatmaps[0].view({1,128,128});
        // cout  << "L470 pre tensorToMat max "<< torch::max(tgt).item<float>()<< endl;

        // cout  << "L470 pre tensorToMat sum "<< tgt.sum({0,1,2}).item<float>() << endl;

        // cout  << "L470 pre tensorToMat full sum "<< jointHeatmaps.sum({0,1,2}).item<float>() << endl;

        // auto mat = tensorToMat(tgt);
        
        // cout  << "L470.5 post tensorToMat single sum "<< cv::sum(mat)[0] << endl;
        // cout  << "L470.5 post tensorToMat single max "<< findMaxValue(mat) << endl;

        // cout  << "L471 "<< mat.size() << endl;
        // cout  << "L472 single sum ,"<< matToTensor(mat).sum({0,1,2}).item<float>() << endl;
        // cout  << "L473 single  2xconv sum ,"<< matToTensor(mat).sum({0,1,2}).item<float>() << endl;
        // cout  << "L474 single 2x stacekd conv sum ,"<<  matToTensor(tensorToMat(matToTensor(mat))).sum({0,1,2}).item<float>()<< endl;

        // saveImageToFile(mat*255, "/scratch/palle.a/AirKeyboard/data/tmp/L462_tmp_heatmap_post_"+std::to_string(i)+".jpg");


        // cout << "GT KP:" << endl;
        // printTensor(shrunk_kp2d);

        //saveImageToFile(tensorToMat(jointHeatmaps[0]*255), "/scratch/palle.a/AirKeyboard/data/tmp/heatmap0_" + std::to_string(i) +".jpg");


        
  
        //cout << jointHeatmaps.sizes() << endl;

        //cout << "input heatmaps:" << endl;
        //printTensor(jointHeatmaps[0]);
        // cout  << "L537 pre kp single sum: "<< jointHeatmaps[0].view({1,128,128}).sum({0,1,2}).item<float>() << endl;

        // auto kp = getKPFromHeatmap(jointHeatmaps, torch::Device(torch::kCPU));
    
        // cout  << "L538 post kp single sum: "<< jointHeatmaps[0].view({1,128,128}).sum({0,1,2}).item<float>() << endl;

        // cout << "extracted KP:" << endl;
        // printTensor(kp);
        
        // auto revImage = tensorToMat(imageTensor);
        // revImage = drawKeypoints(revImage, kp, cv::Scalar(0,0,255));
        // //saveImageToFile(revImage, "/scratch/palle.a/AirKeyboard/data/tmp/draw_kp_" + std::to_string(i) +".jpg");


        // mat = tensorToMat(jointHeatmaps);
        // cout  << mat.size() << endl;
        // saveImageToFile(mat*255, "/scratch/palle.a/AirKeyboard/data/tmp/L484_tmp_heatmap_post_"+std::to_string(i)+".jpg");

        // if (i == 1){
        // cout << "L549 , pre-push sum (full hm)" << jointHeatmaps.sum({0,1,2}).item<float>() << endl;
        // }
        // xData.push_back(imageTensor);
        // yData.push_back(jointHeatmaps);
        xData[i - 1] = imageTensor;
        yData[i - 1] = jointHeatmaps;


    }

    xData = removeNullVectors(xData);
    yData = removeNullVectors(yData);
    torch::Tensor xTensor = torch::stack(xData);
    torch::Tensor yTensor = torch::stack(yData);

    std::string tmpFilePath = "/scratch/palle.a/AirKeyboard/data/tmp/pp_rand_hm.jpg";
    // not an issue with jointToHeatMaps , sum remains the same

    // cout << "L563 " << yTensor.sizes() << endl;
    // auto randomHeatmap = yTensor[0][0].view({1,128,128});

    // cout << "L566: pp_rand sum (single hm)" << randomHeatmap.sum({0,1,2}).item<float>() << endl;
    // saveImageToFile((tensorToMat(randomHeatmap))*255,tmpFilePath);



    Dataset d;
    d.x = xTensor;
    d.y = yTensor;
    return d;

}


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



