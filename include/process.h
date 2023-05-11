#pragma once
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "config.h"
#include <map>

struct AffineMatrix {
    float value[6];
};

void preprocess(uint8_t* src, const int& src_width, const int& src_height,
    float* dst, const int& dst_width, const int& dst_height,
    cudaStream_t stream, float& scale);

void NMS(std::vector<Detection>& res, float* output, const float& conf_thresh, const float& nms_thresh);

void drawBbox(cv::Mat& img, std::vector<Detection>& res, float& scale, std::map<int, std::string>& Labels);
