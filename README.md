
# YOLOv8
The Pytorch implementation is [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

# Reference 
Reference wang-xinyu [tensorrtx](https://github.com/wang-xinyu/tensorrtx)

# Requirements
TensorRT 8.2\
OpenCV 4.6.0

# Config
Choose the model n/s/m/l/x from command line arguments.\
Check more configs in include/config.h

# How to Run, yolov8s as example
1. generate .wts from pytorch with .pt
```
// download https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
cd yolov8
python gen_wts.py
// a file 'yolov8s.wts' will be generated.
```
2. Run CMakeLists.txt with Cmake

3. Open project and run it

# INT8 Quantization
1. set the macro USE_INT8 in config.h and make

2. serialize the model and test

![bus](https://github.com/tianqiang1223/yolov8_tensorrt/blob/main/images/bus.jpg)
