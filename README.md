# Object Detection using mobilenet SSD

OpenCV works only with USB cameras. So for pi we need to use picamera2 module and code is slightly different.

We will have then 2 folders that will have custom `main` code. We also use 2 different models

## Run the program with OPENCV (so on your laptop)

```bash
chmod +x ./opencv/setup.sh && ./opencv/setup.sh
python3 -m venv venv-mobilenet
source venv-mobilenet/bin/activate
pip install opencv-python
cd opencv && python3 main.py # First run might fail due to missing camera permission
```

## Run the program with PICAMERA2 (so on your headless raspi)

```bash
chmod +x ./picamera2/setup.sh && ./picamera2/setup.sh
sudo apt install -y python3-picamera2
sudo apt install -y python3-opencv
python3 -m venv --system-site-packages venv-mobilenet
source venv-mobilenet/bin/activate
cd picamera2 && python3 main.py # First run might fail due to missing camera permission
```

Break with `ESC` or by killing terminal process `CTRL + C` (linux and mac, dunno windows shell).

## What is mobilenet?

Mobilenet is a type of convolutional neural network designed for mobile and embedded vision applications. Instead of using standard convolution layers, they are based on a streamlined architecture that uses depthwise separable convolutions. Using this architecture, we can build lightweight deep neural networks that have low latency for mobile and embedded devices (example: jetson nano).

Read more about the network architecture in the [original paper by Google](https://arxiv.org/abs/1704.04861v1) researchers in 2017.

## Install Requirements

### opencv

For `OPENCV` project we will use pre-trained model weights and model definition from [MobilNet_SSD_opencv](https://github.com/djmv/MobilNet_SSD_opencv)

- **Model definition:** https://github.com/djmv/MobilNet_SSD_opencv/blob/master/MobileNetSSD_deploy.prototxt
- **Pre-trained model weights:** https://github.com/djmv/MobilNet_SSD_opencv/blob/master/MobileNetSSD_deploy.caffemodel

> CAFFE (Convolutional Architecture for Fast Feature Embedding) is a deep learning framework for creating image classification and image segmentation models. Initially, users can create and save their models as plain text PROTOTXT files. After the model is trained and refined using Caffe, the program saves the trained model as a CAFFEMODEL file.

### picamera2

The object detection neural network model and other files used can be downloaded from [Object_Detection_Files](https://core-electronics.com.au/media/kbase/491/Object_Detection_Files.zip)

- The model (MobileNet v3) is trained on the [“Common Objects in Context” - COCO data set](https://cocodataset.org/#home).

> Run `chmod +x ./setup.sh` and `./setup.sh` in order to download them and store in correct folder.
