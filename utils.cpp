#include <vector>
#include <typeinfo>

using namespace std;

void printVector(vector vec){
    for (auto& el : vec) {
        std::cout << el << ", "; 
    }

    return;
}

template <typename T>
void printType(const T& object) {
    std::cout << typeid(object).name() << std::endl;
}

int charToInt(char c){
    return c - '0';
}

