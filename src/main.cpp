#include "model.h"
#include "utils.h"
#include "process.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

Logger gLogger;

void serializeEngine(const int& kBatchSize, std::string& wts_name, std::string& engine_name, std::string& sub_type){
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::IHostMemory* serialized_engine = nullptr;

    if(sub_type == "n"){
        serialized_engine = buildEngineYolov8n(kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
    }
    else if (sub_type == "s") {
        serialized_engine = buildEngineYolov8s(kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
    }
    else if (sub_type == "m") {
        serialized_engine = buildEngineYolov8m(kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
    }
    else if (sub_type == "l") {
        serialized_engine = buildEngineYolov8l(kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
    }
    else if (sub_type == "x") {
        serialized_engine = buildEngineYolov8x(kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
    }

    assert(serialized_engine);
    std::ofstream p(engine_name, std::ios::binary);
    if(!p){
        std::cout << "could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

    delete builder;
    delete config;
    delete serialized_engine;
}


void Inference(std::string& engine_name, std::string& class_file) {
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;

    readEngineFile(engine_name, runtime, engine, context);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

    float* device_buffers[2];
    uint8_t* image_device = nullptr;
    float* output_buffer_host = new float[kBatchSize * kOutputSize];
    assert(engine->getNbBindings() == 2);
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    cudaMalloc((void**)&image_device, kMaxInputImageSize * 3);
    cudaMalloc((void**)&device_buffers[0], kBatchSize * 3 * kInputH * kInputW * sizeof(float));
    cudaMalloc((void**)&device_buffers[1], kBatchSize * kOutputSize * sizeof(float));
    
    std::map<int, std::string> labels;
    readClassFile(class_file, labels);

    cv::Mat image;
    cv::VideoCapture cap(0);
    while (char(cv::waitKey(1) != 27)) {
        auto t_beg = std::chrono::high_resolution_clock::now();
        cap >> image;
        float scale = 1.0;
        int img_size = image.cols * image.rows * 3;
        cudaMemcpyAsync(image_device, image.data, img_size, cudaMemcpyHostToDevice, stream);
        preprocess(image_device, image.cols, image.rows, device_buffers[0], kInputW, kInputH, stream, scale);
        context->enqueue(kBatchSize, (void**)device_buffers, stream, nullptr);
        cudaMemcpyAsync(output_buffer_host, device_buffers[1], kBatchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        std::vector<Detection> res;
        NMS(res, output_buffer_host, kConfThresh, kNmsThresh);
        drawBbox(image, res, scale, labels);
        cv::imshow("Inference", image);
        auto t_end = std::chrono::high_resolution_clock::now();
        float total_inf = std::chrono::duration<float, std::milli>(t_end - t_beg).count();
        std::cout << "Inference time: " << int(total_inf) << std::endl;
    }

    cv::destroyAllWindows();

    // Release stream and buffers
    cudaStreamDestroy(stream);
    cudaFree(device_buffers[0]);
    cudaFree(device_buffers[1]);
    delete[] output_buffer_host;
    // Destroy the engine
    delete context;
    delete engine;
    delete runtime;
} 

int main() {
    std::string wts_name = "../weights/yolov8s.wts";
    wts_name = "";
    std::string engine_name = "../weights/yolov8s.engine";
    std::string class_file = "../weights/classes.txt";
    std::string sub_type = "s";

    if (!wts_name.empty()) {
        serializeEngine(kBatchSize, wts_name, engine_name, sub_type);
        return 0;
    }

    Inference(engine_name, class_file);

    return 0;
}

/****************************************************************************************************
******************************************  dllexport  **********************************************
*****************************************************************************************************/

//extern "C" __declspec(dllexport) void startInference() {
//    std::string engine_name = "./weights/yolov8s.engine";
//    std::string class_file = "./weights/classes80.txt";
//
//    Inference(engine_name, class_file);
//
//    return;
//}

