
#include "data_work.h"
#include "model_utils.h"
#include "train.h"
#include <json.hpp>

using namespace std;

int main(int argc, char* argv[]) {


    if (argc <= 1){
        cout <<"Please provide input file path as first argument"<< std::endl;
        return 0;
    }


    std::string inputFilePath = std::string(argv[1]);


    nlohmann::json inputParams = loadJson(inputFilePath);

    std::string modelPath = inputParams["modelPath"].get<std::string>();
    std::string dataPath = inputParams["dataPath"].get<std::string>();
    std::string saveName = inputParams["saveName"].get<std::string>();
    std::string lossName = inputParams["lossName"].get<std::string>();

    cout << "Running evaluation for model at \n" << modelPath <<
        "\nwith data at \n" << dataPath <<
        "\nwith save name of \n" << saveName << std::endl;

    auto testSaveName = "test_"+saveName;
    auto trainSaveName = "train_"+saveName;
    auto data = loadSamples(dataPath);
    Dataset train = data[0];
    Dataset test = data[1];

    nlohmann::json modelParams = inputParams["modelParams"];
    ModelBuilder* modelBuilder = createModelBuilder(modelParams, train)

    

    Loss* loss = getLoss(lossName);


    TrainParams tp =  TrainParams(loss, modelBuilder);
    tp.setModelPath(modelPath);
    tp.setModelParams(modelParams);

    evaluate(test, tp, testSaveName, true);

    evaluate(train, tp, trainSaveName, true);

}