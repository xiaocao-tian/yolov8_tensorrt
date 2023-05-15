cmake_minimum_required(VERSION 3.10)

project(yolov8)

# Modify to your path
set(OpenCV_DIR "D:/software/opencv/build")  
set(TRT_DIR "D:/software/TensorRT-8.5.2.2")  

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# setup CUDA
find_package(CUDA REQUIRED)
message(STATUS "libraries: ${CUDA_LIBRARIES}")
message(STATUS "include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})

enable_language(CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/plugin)

# TensorRT
find_package(TRT_DIR)
include_directories(${TRT_DIR}/include)
link_directories(${TRT_DIR}/lib)

# OpenCV
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
add_executable(yolov8 ${PROJECT_SOURCE_DIR}/src/main.cpp ${SRCS})

target_link_libraries(yolov8 nvinfer)
target_link_libraries(yolov8 cudart)
target_link_libraries(yolov8 myplugins)
target_link_libraries(yolov8 ${OpenCV_LIBS})

