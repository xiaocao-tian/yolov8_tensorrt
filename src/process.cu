#include "process.h"

__global__ void warpaffine_kernel(
	uint8_t* src, int src_line_size, int src_width,
	int src_height, float* dst, int dst_width,
	int dst_height, uint8_t const_value_st,
	AffineMatrix d2s, int edge) {
	int position = blockDim.x * blockIdx.x + threadIdx.x;
	if (position >= edge) return;

	float m_x1 = d2s.value[0];
	float m_y1 = d2s.value[1];
	float m_x2 = d2s.value[3];
	float m_y2 = d2s.value[4];

	int dx = position % dst_width;
	int dy = position / dst_width;
	float src_x = m_x1 * dx + m_y1 * dy;
	float src_y = m_x2 * dx + m_y2 * dy;
	float c0, c1, c2;

	if (src_x < 0 || src_x + 1 >= src_width || src_y < 0 || src_y + 1 >= src_height) {
		c0 = const_value_st;
		c1 = const_value_st;
		c2 = const_value_st;
	}
	else {
		int x_low = floorf(src_x);
		int y_low = floorf(src_y);
		int x_high = x_low + 1;
		int y_high = y_low + 1;
		float w1 = (y_high - src_y) * (x_high - src_x);
		float w2 = (y_high - src_y) * (src_x - x_low);
		float w3 = (src_y - y_low) * (x_high - src_x);
		float w4 = (src_y - y_low) * (src_x - x_low);
		uint8_t* v1 = src + y_low * src_line_size + x_low * 3;
		uint8_t* v2 = src + y_low * src_line_size + x_high * 3;
		uint8_t* v3 = src + y_high * src_line_size + x_low * 3;
		uint8_t* v4 = src + y_high * src_line_size + x_high * 3;
		c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
		c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
		c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
	}

	// bgr -> rgb
	float temp = c2;
	c2 = c0;
	c0 = temp;

	// normalization
	c0 /= 255.0f;
	c1 /= 255.0f;
	c2 /= 255.0f;

	// rgbrgbrgb -> rrrgggbbb
	int area = dst_height * dst_width;
	float* pdst_c0 = dst + dy * dst_width + dx;
	float* pdst_c1 = pdst_c0 + area;
	float* pdst_c2 = pdst_c1 + area;
	*pdst_c0 = c0;
	*pdst_c1 = c1;
	*pdst_c2 = c2;
}

void preprocess(
	uint8_t* src, const int& src_width, const int& src_height,
	float* dst, const int& dst_width, const int& dst_height,
	cudaStream_t stream, float& scale) {

	AffineMatrix s2d, d2s;
	scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);
	s2d.value[0] = scale;
	s2d.value[1] = 0;
	s2d.value[2] = 0;
	s2d.value[3] = 0;
	s2d.value[4] = scale;
	s2d.value[5] = 0;
	cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
	cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
	cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

	memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

	int jobs = dst_height * dst_width;
	int threads = 256;
	int blocks = ceil(jobs / (float)threads);
	warpaffine_kernel << <blocks, threads, 0, stream >> > (
		src, src_width * 3, src_width,
		src_height, dst, dst_width,
		dst_height, 128, d2s, jobs);
}


static float iou(float lbox[4], float rbox[4]) {
	float interBox[] = {
	  (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
	  (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
	  (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
	  (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
	};

	if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
		return 0.0f;

	float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
	return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

static bool cmp(const Detection& a, const Detection& b) {
	return a.conf > b.conf;
}


void NMS(std::vector<Detection>& res, float* output, const float& conf_thresh, const float& nms_thresh) {
	int det_size = sizeof(Detection) / sizeof(float);
	std::map<float, std::vector<Detection>> m;
	for (int i = 0; i < output[0]; i++) {
		if (output[1 + det_size * i + 4] <= conf_thresh) continue;
		Detection det;
		memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
		if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
		m[det.class_id].push_back(det);
	}
	for (auto it = m.begin(); it != m.end(); it++) {
		auto& dets = it->second;
		std::sort(dets.begin(), dets.end(), cmp);
		for (size_t m = 0; m < dets.size(); ++m) {
			auto& item = dets[m];
			res.push_back(item);
			for (size_t n = m + 1; n < dets.size(); ++n) {
				if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
					dets.erase(dets.begin() + n);
					--n;
				}
			}
		}
	}
}

//cv::Rect getRect(cv::Mat& img, float bbox[4], float& scale) {
//	float l, r, t, b;
//	l = bbox[0] / scale;
//	t = bbox[1] / scale;
//	r = bbox[2] / scale;
//	b = bbox[3] / scale;
//	return cv::Rect(int(l), int(t), int(r - l), int(b - t));
//}
//
//void drawBbox(cv::Mat& img, std::vector<Detection>& res, float& scale, std::map<int, std::string>& Labels) {
//	for (size_t j = 0; j < res.size(); j++) {
//		cv::Rect r = getRect(img, res[j].bbox, scale);
//		std::string name = Labels[(int)res[j].class_id];
//		cv::rectangle(img, r, cv::Scalar(0xFF, 0xFF, 0), 2);
//		cv::putText(img, name, cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0), 2);
//	}
//}

void drawBbox(cv::Mat& img, std::vector<Detection>& res, float& scale, std::map<int, std::string>& Labels) {
	for (size_t j = 0; j < res.size(); j++) {
		float l = res[j].bbox[0] / scale;
		float t = res[j].bbox[1] / scale;
		float r = res[j].bbox[2] / scale;
		float b = res[j].bbox[3] / scale;
		cv::Rect rect = cv::Rect(int(l), int(t), int(r - l), int(b - t));
		std::string name = Labels[(int)res[j].class_id];
		cv::rectangle(img, rect, cv::Scalar(0xFF, 0xFF, 0), 2);
		cv::putText(img, name, cv::Point(rect.x, rect.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0), 2);
	}
}
