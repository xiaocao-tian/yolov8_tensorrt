#include <fstream>
#include <io.h>
#include "calibrator.h"


cv::Mat Calibrator::preprocess_img(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    }
    else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}


void Calibrator::getFiles(std::string path, std::vector<std::string>& files) {
    intptr_t Handle;
    struct _finddata_t FileInfo;
    std::string p;
    Handle = _findfirst(p.assign(path).append("\\*").c_str(), &FileInfo);

    while (_findnext(Handle, &FileInfo) == 0) {
        if (strcmp(FileInfo.name, ".") != 0 && strcmp(FileInfo.name, "..") != 0) {
            files.push_back(FileInfo.name);
        }
    }
}


Calibrator::Calibrator(int batchsize, int input_w, int input_h, std::string img_dir, const char* calib_table_name, bool read_cache) {
    BATCHSIZE = batchsize;
    WIDTH = input_w;
    HEIGHT = input_h;
    INDEX = 0;
    IMAGEDIR = img_dir;
    CALIBRATORTABLE = calib_table_name;
    READCACHE = read_cache;
    INPUTSIZE = BATCHSIZE * 3 * WIDTH * HEIGHT;

    cudaMalloc(&DEVICEINPUT, INPUTSIZE * sizeof(float));
    getFiles(IMAGEDIR, IMAGEFILES);
}


Calibrator::~Calibrator() {
    cudaFree(DEVICEINPUT);
}


int Calibrator::getBatchSize() const noexcept {
    return BATCHSIZE;
}


bool Calibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
    if (INDEX + BATCHSIZE > (int)IMAGEFILES.size()) return false;

    std::vector<cv::Mat> input_imgs;
    for (int i = INDEX; i < INDEX + BATCHSIZE; i++) {
        cv::Mat temp = cv::imread(IMAGEDIR + IMAGEFILES[i]);
        if (temp.empty()) {
            std::cerr << "Image cannot open!" << std::endl;
            return false;
        }
        cv::Mat pr_img = preprocess_img(temp, WIDTH, HEIGHT);
        input_imgs.push_back(pr_img);
    }
    INDEX += BATCHSIZE;
    cv::Mat blob = cv::dnn::blobFromImages(input_imgs, 1.0 / 255.0, cv::Size(WIDTH, HEIGHT), cv::Scalar(0, 0, 0), true, false);

    cudaMemcpy(DEVICEINPUT, blob.ptr<float>(0), INPUTSIZE * sizeof(float), cudaMemcpyHostToDevice);
    bindings[0] = DEVICEINPUT;
    return true;
}


const void* Calibrator::readCalibrationCache(size_t& length) noexcept {
    std::cout << "reading calib cache: " << CALIBRATORTABLE << std::endl;
    CALIBRATORCACHE.clear();
    std::ifstream input(CALIBRATORTABLE, std::ios::binary);
    input >> std::noskipws;
    if (READCACHE && input.good()) {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(CALIBRATORCACHE));
    }
    length = CALIBRATORCACHE.size();
    return length ? CALIBRATORCACHE.data() : nullptr;
}


void Calibrator::writeCalibrationCache(const void* cache, size_t length) noexcept {
    std::cout << "writing calib cache: " << CALIBRATORTABLE << std::endl;
    std::ofstream output(CALIBRATORTABLE, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}
