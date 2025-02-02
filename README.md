# Object Detection using mobilenet SSD (WIP)

OpenCV works only with USB cameras. So for pi we need to use picamera2 module and code is slightly different.

Had to develop 2 solutions then, in order to test locally and on pi hardware eventually.
Thus, for the time being we have then 2 folders that have custom `main` code. We also refer to the Caffe MobileNetV3 model and the SSD MobileNetV3 model as two variations or different versions of the MobileNetV3 model architecture. More on this below.

## Set enviroment variables

```bash
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=7794819465:AAHQmG6TdrWdRMYrIMSOCcEwiOZtdhlSuaA
TELEGRAM_CHAT_ID=-4661983422
NOTIFICATION_COOLDOWN=60
```

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

- **Caffe MobileNetV3 Model:** This refers to a specific implementation or version of the MobileNetV3 architecture trained using the Caffe deep learning framework. Caffe is a deep learning framework developed by Berkeley AI Research (BAIR) and is known for its efficiency in training and deploying deep neural networks. The MobileNetV3 model implemented in Caffe is likely trained for specific tasks such as image classification or feature extraction.

- **SSD MobileNetV3 Large COCO Model:** This model, on the other hand, is an implementation of the MobileNetV3 architecture trained specifically for object detection tasks using the Single Shot Multibox Detector (SSD) framework. It is trained on the COCO (Common Objects in Context) dataset, which contains various object categories along with bounding box annotations.

The main difference between these two variations lies in their architecture and the specific tasks they are designed for:

The `Caffe MobileNetV3` model is likely optimized for image classification tasks, where the goal is to classify an input image into one of several predefined categories.
The `SSD MobileNetV3` model, being trained for object detection, is capable of not only classifying objects in an image but also localizing them by providing bounding box coordinates.

In summary, while both variations are based on the `MobileNetV3` architecture, they are trained for different tasks and may have differences in their network architecture, training data, and performance characteristics.

Nevertheless, we use them to accomplish same task ;)

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
