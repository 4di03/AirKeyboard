cmake_minimum_required(VERSION 3.18)
project( Open_CV_Project )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH "/Users/adithyapalle/PyTorch/pytorch-install")

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( Open_CV_Project main.cpp )
target_link_libraries(Open_CV_Project "${TORCH_LIBRARIES}")
target_link_libraries( Open_CV_Project  ${OpenCV_LIBS})
set_property(TARGET Open_CV_Project PROPERTY CXX_STANDARD 17)


