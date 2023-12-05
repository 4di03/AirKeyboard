cmake_minimum_required(VERSION 3.18) # FOR LINUX
project( Open_CV_Project )


# Set the path to your CUDA installation
set(CUDA_TOOLKIT_ROOT_DIR /shared/centos7/cuda/11.8)

# Set C++ standard to 17
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable C++11
#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Add necessary flags for GCC 11.1.0
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif()


#set(CMAKE_OSX_ARCHITECTURES x86_64)


list(APPEND CMAKE_PREFIX_PATH "/home/palle.a/CPP_packages")


# Now you can find and use the xtensor, xtensor-blas, and xtensor-io libraries
find_package(xtensor REQUIRED)


# Add the path to the OpenCVConfig.cmake file in the build directory
#set(OpenCV_DIR "/home/palle.a/CPP_packages/opencv_build")

find_package(OpenCV REQUIRED)
list(APPEND CMAKE_PREFIX_PATH "/home/palle.a/CPP_packages/libtorch")

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${CUDA_INCLUDE_DIRS})
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(/home/palle.a/CPP_packages/json/single_include/nlohmann)

add_executable( Open_CV_Project 
        main.cpp  
        data_work.cpp
        train.cpp
        train.h
        cunet/cunet.h)
add_executable( Test test.cpp data_work.cpp)

target_link_libraries(Test ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries(Test "${TORCH_LIBRARIES}")
target_link_libraries(Test xtensor)
target_link_libraries(Test ${OpenCV_LIBS})


target_link_libraries(Open_CV_Project ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries(Open_CV_Project "${TORCH_LIBRARIES}")
target_link_libraries( Open_CV_Project  ${OpenCV_LIBS})
target_link_libraries(Open_CV_Project xtensor)
target_link_libraries(Open_CV_Project opencv_imgcodecs)


set_property(TARGET Open_CV_Project PROPERTY CXX_STANDARD 17)


