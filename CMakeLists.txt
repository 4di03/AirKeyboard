cmake_minimum_required(VERSION 3.18)
project( Open_CV_Project )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_OSX_ARCHITECTURES x86_64)


list(APPEND CMAKE_PREFIX_PATH "/Users/adithyapalle/CPP_packages")


# Now you can find and use the xtensor, xtensor-blas, and xtensor-io libraries
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(OpenCV REQUIRED)
list(APPEND CMAKE_PREFIX_PATH "/Users/adithyapalle/CPP_packages/pytorch-install")

find_package(Torch REQUIRED)



#find_package(pybind11 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(/Users/adithyapalle/CPP_packages/json/single_include/nlohmann)

add_executable( Open_CV_Project main.cpp data_work.cpp
        train.cpp
        train.h)

target_link_libraries(Open_CV_Project "${TORCH_LIBRARIES}")
target_link_libraries( Open_CV_Project  ${OpenCV_LIBS})
target_link_libraries(Open_CV_Project xtensor)
target_link_libraries(Open_CV_Project xtensor-blas)
set_property(TARGET Open_CV_Project PROPERTY CXX_STANDARD 17)


