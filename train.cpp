#include <iostream>
#include <torch/torch.h>
#include "data_work.h"
#include <cmath>
#include <vector>
#include <torch/script.h>
#include "train.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iterator>
#include <fstream>
#include "cunet/cunet.h"
#include "model.h"

using namespace std;







torch::Tensor removeExtraDim(torch::Tensor x){
    std::vector<int64_t> sizes;
    
    for (int64_t size : x.sizes()){
        if (size != 1){
            sizes.push_back(size);
        }
    }

    return x.view(sizes);
}
template <typename num>
float getMean(std::vector<num> nums){
    float sum = 0;

    for (float n :nums){
        sum += n;
    }

    return sum/nums.size();

}


torch::Device initDevice(bool cuda){
    torch::Device device(torch::kCUDA,0);
    if (!cuda){
        cout << "using CPU" << endl;
        device = torch::Device(torch::kCPU);
    }else{
        cout <<"USING CUDA" << endl;
    }

    return device;
}


// Load Model
void loadModel(const std::string& model_path, torch::nn::Module* model) {
    //torch::jit::script::Module model;

    try {
        //model = torch::load(model,model_path)
        torch::serialize::InputArchive input_archive;
        input_archive.load_from(model_path);
        model->load(input_archive);
        //model = torch::jit::load(model_path);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
    }

}


void printModel(torch::nn::Module model) {
    // Get named parameters of the model, gets parameters of last layer
    auto named_params = model.named_parameters();
    cout << "MODEL PARAMS (last layer): " << endl;
    // Iterate over named parameters and print their names and values
    auto& named_param = named_params[named_params.size()-1];
    const std::string& name = named_param.key();
    const torch::Tensor& param = named_param.value();
    std::cout << "Parameter Name: " << name;
    std::cout << " Parameter Value:\n" << param;
    std::cout << "------------------------" << std::endl;
    
}

float evaluateTest( Dataset test, torch::Device device, Model& model, Loss& loss_fn){
    //auto loss_fn = MSELoss();

    model.to(device);
    //printModel(model);
    //test = test.sample(0.025);
    test.y = removeExtraDim(test.y);
    model.eval();

    float maxBatchSize = 128.0;

    int nTrainSamples = test.y.sizes()[0];
    int nBatches = ceil(nTrainSamples / maxBatchSize);

    //cout << "Breaking data into " << nBatches << " Batches" << endl;

    float totalLoss = 0;

    for (int i=0 ; i < nBatches ; i++){
        // Get the batch
        auto batch = test.getBatch(i, maxBatchSize);
        auto x = batch.first;
        auto y = batch.second;

        x = x.to(device);

        auto pred = model.forward(x);//model.forward(x);

        // free up gpu space
        x.reset();
        
        y = y.to(device);

        //cout << pred.sizes() << endl;
        torch::Tensor loss = loss_fn.forward(pred, y) * y.sizes()[0];

        //cout << loss.item<float>() << endl;
        totalLoss += loss.item<float>();

    }


    //cout << "L411 nTrainSamples: " << nTrainSamples << ",maxBatchSize: " << maxBatchSize << ",nBatches: " << nBatches << ", total_loss: " << totalLoss << endl;
    float mse = totalLoss/nTrainSamples;

    cout <<"Loss: " <<  mse<< endl;

    return mse;

}



void drawPredictions(Dataset d,Model& model, const std::string& valLossSavePath , torch::Device device){
    // d should be small enough for the model to produce infereence in a single batch
    d.to(device);
    

    //auto rawImages = dataset.x;
    auto imageData = d.x;
    auto expectedHeatmaps = d.y;



    //auto pred =model.forward(imageData); //model.forward(imageData);

    int nImages = d.x.sizes()[0];
    auto gtColor = cv::Scalar(0, 0, 255); // in BGR
    auto predColor = cv::Scalar(0,255,0); // in BGR

    cout << "Ground truth keypoints are in Red, predicted are in Green" << endl;

    std::filesystem::create_directories(valLossSavePath);
    auto mse = MSELoss();
    auto iou = IouLoss();

    std::vector<int> predTimes; // in microseconds

    for (int i = 0; i< nImages; i++){

        auto start_time = std::chrono::high_resolution_clock::now();

        auto it = imageData[i].unsqueeze(0);
       // cout << "L186: " << torch::max(it) << endl; # is normalized
        auto predMap = model.forward(it);
        auto predKP = getKPFromHeatmap(predMap,device);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        // Convert duration to int
        int duration_int = static_cast<int>(duration.count());
        predTimes.push_back(duration_int);
   

        auto expectedKP = getKPFromHeatmap(expectedHeatmaps[i],device);

        auto predHM = predMap;//pred[i].unsqueeze(0);
        auto expHM = expectedHeatmaps[i].unsqueeze(0);



        // cv::Mat img1;
        // imageData[i].convertTo(img1, CV_8U, 255, 0);

       // cout << "L206(pre-tensorToMat) Max: " <<torch::max(imageData[i]) <<  " Mean: " << torch::mean(imageData[i]) << endl;

        cv::Mat image = tensorToMat(imageData[i]); // draw on non-standardized image

        cv::Mat img2;
        image.convertTo(img2, CV_8U, 255, 0);

        cv::imwrite("L207_image_convert_img2.jpg" , img2);

        img2 = drawKeypoints(img2 ,expectedKP, gtColor);

        img2 = drawKeypoints(img2 ,predKP, predColor);  


        std::string filePath = valLossSavePath + "Image" + std::to_string(i) + ".jpg";
        saveImageToFile(img2, filePath);



    }
    cout << "Saved all predictions to " + valLossSavePath << endl;
    cout << "Prediction per image took " << getMean(predTimes) << " microseconds  on average" << endl;

    return;

}

void evaluate(Dataset& test,  TrainParams t ,bool draw = true){ // need to pass abstract classes by reference
/**
bool cuda, std::string model_name,bool draw, Loss& loss_fn
*/
    auto cuda = t.cuda;
    auto model_name = t.model_name;
    auto loss = t.loss_fn;
    auto device = initDevice(cuda);

    // cout << "running evaluation for " << model_name << endl;

    Model* model;

    loadModel("/scratch/palle.a/PalmPilot/data/models/" + model_name, model);


    // cout << "L248: " ;
    // Model* model = new JitModel("/scratch/palle.a/PalmPilot/python_sample/weights/model_final.pt", device);

    evaluateTest(test,device,*model,*loss);
    

    if (draw){
    auto valLossSavePath = "/scratch/palle.a/PalmPilot/data/analytics/" + model_name+ "_analytics/";
    drawPredictions(test.slice(10)[0], *model, valLossSavePath , device); // draws first  10 images in test set
    }

}


template <typename T>
void writeVectorToFile(const std::vector<T>& vec, const std::string& filename) {
    const std::string& delimiter = "\n";

    // Create parent directories if they do not exist
    std::filesystem::create_directories(std::filesystem::path(filename).parent_path());


    // Open the file for writing
    std::ofstream outputFile(filename);
    
    // Check if the file is open
    if (!outputFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Use an iterator to copy vector elements to the output stream
    std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<T>(outputFile, delimiter.c_str()));

    // Write the last element without the delimiter
    outputFile << vec.back();

    // Close the file
    outputFile.close();
}


class LossTracker {
public:
    int noImprovementCount = 0;
    int patience;
    float bestLoss = 100000000.0;
    LossTracker( int patience):patience(patience){}

    // Early stopping logic based on validation loss
    bool earlyStopping(float validationLoss) {

        if (validationLoss < bestLoss) {
            bestLoss = validationLoss;
            noImprovementCount = 0;
        } else {
            noImprovementCount++;
        }


        return noImprovementCount >= patience;
    }


};






void trainModel(Dataset& train, 
                Dataset& test, 

                TrainParams tp) {
    /**
     * @brief This function trains the model
     * @param train: Dataset
     * @param test: Dataset
     * ASSUMES CUDA WITH GPU IS AVAILABLE.
     */

    auto loss_fn = tp.loss_fn;

    bool cuda = tp.cuda; 
    torch::Device device = initDevice(cuda);

    float propDataUsed  = tp.propDataUsed;
    std::string model_name = tp.model_name;
    cout << "train x shape: " << train.x.sizes() << endl;
    cout << "train y shape: " << train.y.sizes() << endl;
    // cout << "val x shape: " << val.y.sizes() << endl;
    // cout << "val y shape: " << val.y.sizes() << endl;
    cout << "test x shape: " << test.x.sizes() << endl;
    cout << "test y shape: " << test.y.sizes() << endl;

    int plateauPatience = 20;


    train = train.sample(propDataUsed);

    auto sets = train.sliceProp(tp.propVal); // use 20% of train data for validation(early stopping)
    Dataset val = sets[0];
    train = sets[1];
    test = test.sample(propDataUsed);



    cout << "POST SAMPLING" << endl;
    cout << "train x shape: " << train.x.sizes() << endl;
    cout << "train y shape: " << train.y.sizes() << endl;
    cout << "val x shape: " << val.y.sizes() << endl;
    cout << "val y shape: " << val.y.sizes() << endl;
    cout << "test x shape: " << test.x.sizes() << endl;
    cout << "test y shape: " << test.y.sizes() << endl;

    // Dataset standardTrainData;
    // Dataset standardValData;
    // Dataset standardTestData;
    StandardizeTransform transformer;
    if(tp.standardize){
        auto trainStats = getNormParams(train.x);
        torch::Tensor trainMeans = std::get<0>(trainStats).to(device);
        torch::Tensor trainStds = std::get<1>(trainStats).to(device);

        cout << "Initializing trian standardizer with below means and sts: " << endl;
        printTensor(trainMeans);
        printTensor(trainStds);


        transformer = StandardizeTransform(trainMeans, trainStds);
    } 
   

    
    auto sizes =train.x.sizes();

    int c = sizes[1];
    int nTrainSamples = sizes[0];
    int initNeurons = tp.initNeurons;
    float batchSize = tp.batchSize;
    int nEpochs = tp.nEpochs;
    int levels = tp.levels;



    Model* model= new ResidualUNet(c, initNeurons, 7);//CuNet(c,21, initNeurons);//


    if (tp.pretrainedModelReady()){
        cout << "Loading pretrained model for retraining" << endl;
        loadModel(tp.modelPath, model);
    }
    
    //Model* model= new JitModel("/scratch/palle.a/PalmPilot/python_sample/weights/untrained_test_model.pt", device);

    if (tp.standardize){
        cout<< "STANDARDIZING DATA" <<endl;
        model->setTransformer(transformer); // 
    }else{
        cout << "NOT STANDARDIZING DATA" <<endl;
    }

    model->to(device);

    printModuleDevice(*model);

    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.1));//.momentum(0.9)); // lr = 1, momentum = 0.9

    torch::optim::StepLR scheduler(optimizer, /* step_size = */ 1, /* gamma = */ 0.5); // do it on plateau ()

    LossTracker plateauTracker(10); // checks for plteaus with 20 patience
    LossTracker lossTracker(50); // checks for when to stop trianig wieht early stopping (50 patitence)
    // ReduceLROnPlateau scheduler(
    //     optimizer, 
    //     0.5,
    //     10,
    //     0.00001
    // );
    std::string model_path = "/scratch/palle.a/PalmPilot/data/models/" + model_name;

    std::vector<float> trainLosses;
    std::vector<float> valLosses;

    //auto standardTrainData = transformer.transform(train);
   //auto standardValData = transformer.transform(val);
    for (size_t epoch = 1; epoch <= nEpochs; ++epoch) {
        // Iterate over batches
        //std::cout << "Epoch: " << epoch << std::endl;

        int nBatches = ceil(nTrainSamples / batchSize);
        //std::vector<float> batchLosses;
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < nBatches; ++i) {
            // Get the batch
            auto batch = train.getBatch(i, batchSize);
            auto x = batch.first;
            auto y = batch.second;



            // std::string tmpFilePath = "/scratch/palle.a/PalmPilot/data/tmp/train_hm_" + std::to_string(i) + ".jpg";
            //cout << "L592" << y.sizes() << endl;
            auto tgt = y.index({0,0}).view({1,128,128});
            //cout << "L593" << tgt.sizes() << endl;
            saveImageToFile(tensorToMat(tgt)*255,"L437_hm_train.jpg");

            // Zero gradients
            optimizer.zero_grad();


            //cout << "x shape: " << x.sizes() << endl;
            // Forward pass
            x = x.to(device);

            torch::Tensor y_pred = model->forward(x);

            y = y.to(device);
            y = removeExtraDim(y); // to remove extra axis for 1 channel image

            // cout << "y_pred shape: " << y_pred.sizes() << endl;
            // cout << "y shape: " << y.sizes() << endl;
            // Compute Loss

            //cout << i << endl;


            torch::Tensor loss = loss_fn->forward(y_pred, y);
            //batchLosses.push_back(loss.item<float>());

            if (i == 0){
                auto bl = loss.item<float>();
                cout << "Train (single batch) Loss " <<  bl<<endl;
                trainLosses.push_back(bl);

            }

            // Backward pass
            loss.backward();

            // Update the parameters
            optimizer.step();
            //cout << i << endl;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        cout << "VALIDATION LOSS: ";
        float valLoss = evaluateTest(val,device,*model, *loss_fn);

        if (plateauTracker.earlyStopping(valLoss)){
            cout << "Reducing Learning Rate!" << endl;
            scheduler.step(); // if no imporvement for n1 epcosh, update LR
        }

        if (lossTracker.earlyStopping(valLoss)){//if loss doesnt imporve for n2 epochs, break out of training
            std::cout << "Early stopping at epoch " << epoch << std::endl;
            
            break;

        }

        valLosses.push_back(valLoss);

        if(epoch % 10 ==0){
            model->save(model_path);
        }
        // Print the time taken for the forward pass
        //std::cout << "Epoch took " << duration.count() << " microseconds." << std::endl;


        //cout << "Epoch Mean Loss " << getMean(batchLosses) <<endl;


    }
    model->save(model_path);

    model->eval();
    torch::NoGradGuard no_grad;


    std::string valLossSavePath = "/scratch/palle.a/PalmPilot/data/analytics/" + model_name + "_analytics";


    writeVectorToFile(trainLosses, valLossSavePath+"/train_loss.list");
    writeVectorToFile(valLosses, valLossSavePath+"/val_loss.list");


    cout << "Post-training loss:";

    evaluateTest(test, device, *model, *loss_fn);


    Dataset sampleTestImages = test.shuffle().slice(10)[0];


    // SAVE MODEL


    drawPredictions(sampleTestImages,*model, valLossSavePath+"/predictions_presave/",device);


    cout << "Loading model after saving from " << model_path << endl ;
    Model* postModel = new ResidualUNet(c, initNeurons,7);//JitModel(model_path,device);
    loadModel(model_path,postModel);
    postModel->eval();
    drawPredictions(sampleTestImages, *postModel, valLossSavePath+"/predictions_postsave/",device);


}

