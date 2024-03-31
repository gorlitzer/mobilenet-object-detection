# Object Detection using mobilenet SSD

## Run the program

```bash
chmod +x ./setup.sh
mkdir models && cd models
mkdir MobileNetSSD && cd ../../
./setup.sh
python3 -m venv venv-mobilenet
source venv-mobilenet/bin/activate
pip install opencv-python
python3 main.py # First run might fail due to missing camera permission
```

Break with `ESC` or by killing terminal process `CTRL + C` (linux and mac, dunno windows shell).

## What is mobilenet?

Mobilenet is a type of convolutional neural network designed for mobile and embedded vision applications. Instead of using standard convolution layers, they are based on a streamlined architecture that uses depthwise separable convolutions. Using this architecture, we can build lightweight deep neural networks that have low latency for mobile and embedded devices (example: jetson nano).

Read more about the network architecture in the [original paper by Google](https://arxiv.org/abs/1704.04861v1) researchers in 2017.

## Install Requirements

For this project we will use pre-trained model weights and model definition from [MobilNet_SSD_opencv](https://github.com/djmv/MobilNet_SSD_opencv)

- **Model definition:** https://github.com/djmv/MobilNet_SSD_opencv/blob/master/MobileNetSSD_deploy.prototxt
- **Pre-trained model weights:** https://github.com/djmv/MobilNet_SSD_opencv/blob/master/MobileNetSSD_deploy.caffemodel

Run `chmod +x ./setup.sh` and `./setup.sh` in order to download them and store in correct folder.

> CAFFE (Convolutional Architecture for Fast Feature Embedding) is a deep learning framework for creating image classification and image segmentation models. Initially, users can create and save their models as plain text PROTOTXT files. After the model is trained and refined using Caffe, the program saves the trained model as a CAFFEMODEL file.

### What's going on

After downloading the above files to our working directory, we need to load the Caffe model using the `OpenCV DNN` function `cv2.dnn.readNetFromCaffe`. Then, we define the class labels on which the network was trained (i.e. COCO labels).

Our model was trained on **21 object classes** which are passed as a dictionary where each key represents the class ID and the respective value is the name of the label.

### Set up the camera

Since in this example we are using a camera feed for object detection, we instantiate an object of `theVideoCapture` class from the OpenCV library. As an input, the `VideoCapture` class receives an index of the device we want to use. If we have a single camera connected to the computer, we pass a value of `0`.

### Format the input

Then, we get the height and width of the image from the camera frame that will be used later to draw bounding boxes around the detected object. Now, we need to transform the image into a `blob` (which is a `4D NumPy array object` — images, channels, width, height) using `cv2.dnn.blobFromImage` function.

This is required to prepare the input image in the required format for the model intake. The input parameters of this function depend on the model that is being used.

- More about [cv2.dnn.blobFromImage](https://pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/)

### Object detection and visualization

The blob object is then set as input to the `net` network followed by a forward pass through the mobilenet network. Now, we loop over the detections — the detection summary is an array in the format 1, 1, N, 7 where N is the number of detected bounding boxes. Each detection has the format `[image_id, label, conf, x_min, y_min, x_max, y_max]`

```
image_id: ID of the image in the batch
label: predicted class ID
conf: confidence score of the predicted class
x_min, y_min: coordinates of the top left bounding box corner
x_max, y_max: coordinates of the bottom right bounding box corner

Note: Coordinates are in normalized format, in range [0, 1]
```

Next, we extract the confidence score of the detected object(s) from the third element `detections[0, 0, i, 2]` in the detection array. If the confidence score of the detected class is greater than the threshold confidence level (to filter out weak predictions), we get the class id of the detected class from the second element `detections[0, 0, i, 1]` in the detection array.

Once the object is detected, we now try to visualize it by drawing a bounding box and adding the label of that object. The detection array returns the normalized top left and bottom right corner coordinates which are scaled to the frame dimension by multiplying with the width and height of the captured frame from the camera. Then, we draw a bounding box around the detected object using the `cv2.rectangle` function. If the detected class id matches with one of the `21 labels mentioned in the classNames` dictionary, we add a text with the name of the label and add a box around the text.
