#pragma once

#include <opencv2/opencv.hpp>
class Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
	Calibrator(int batchsize, int input_w, int input_h, std::string img_dir, const char* calib_table_name, bool read_cache = true);

	virtual ~Calibrator();
	int getBatchSize() const noexcept override;
	bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
	const void* readCalibrationCache(size_t& length) noexcept override;
	void writeCalibrationCache(const void* cache, size_t length) noexcept override;

private:
	int BATCHSIZE;
	int WIDTH;
	int HEIGHT;
	int INDEX;
	std::string IMAGEDIR;
	std::vector<std::string> IMAGEFILES;
	size_t INPUTSIZE;
	std::string CALIBRATORTABLE;
	bool READCACHE;
	void* DEVICEINPUT;
	std::vector<char> CALIBRATORCACHE;

	cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h);
	void getFiles(std::string path, std::vector<std::string>& files);
};
