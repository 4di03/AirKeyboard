#include <vector>
using namespace std;

void printVector(vector vec){
    for (auto& el : vec) {
        std::cout << el << ", "; 
    }

    return;
}