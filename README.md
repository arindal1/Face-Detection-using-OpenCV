# Face Detection using OpenCV
### *This is a personal project, a Face Detection algorithm in Python using OpneCV*

## What is OpenCV?
- OpenCV is a library of programming functions mainly for real-time computer vision. Originally developed by Intel, it was later supported by Willow Garage, then Itseez. The library is cross-platform and licensed as free and open-source software under Apache License.
  [[Wikipedia]](https://en.wikipedia.org/wiki/OpenCV)

## What is face Detection?
- Face recognition is a method of verifying the identity of a person using their face.
- Deep learning is super popular for face recognition applications.

## Face recognition using Deep Learning:
```
Training a complex network required here will take a significant amount of data and computation power.
A pre-trained network trained by Davis King on a dataset of ~3 million images is used to speed up the process.
The network outputs a vector of 128 numbers which represent the most important features of a face.
```

- **Step 1:** Face Detection
  - The exact location/coordinatesof face is extracted from media.
- **Step 2:** Feature Extraction
  - Face embedding is used with each face to convert it into a vectorand this technique is called Deep Metric Learning.
- **Step 3:** Training a neural network
  - A neural network may output faces which look very similar to each other.
- **Step 4:** Feature map across the face
  - After training the network, it understands to group similar looking facestogether into one category.
- **Step 5:** Embeddings for images are obtained after training
<br>
<br>

---
---
# Old Version:

This script detects faces in an input image and draws rectangles around the detected faces. Here's a breakdown of the code:

1. Importing OpenCV: Import the OpenCV library using `import cv2`.

2. Loading the Pre-trained Classifier: The `CascadeClassifier` class from OpenCV is used to load the pre-trained classifier for face detection. The classifier XML file should be in the same directory as your script. The line `face_cascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')` loads the classifier.

3. Reading the Input Image: The input image, named `test3.jpg`, is read using the `cv2.imread` function.

4. Face Detection: The `detectMultiScale` function is used to detect faces in the input image. It takes the input image, a scale factor, and a minimum number of neighbors as parameters. The `faces` variable will store the coordinates of the detected faces.

5. Drawing Rectangles: A loop iterates through the detected faces (stored as `(x, y, w, h)`), where `(x, y)` is the top-left corner of the face and `(w, h)` is its width and height. For each detected face, a rectangle is drawn around it using the `cv2.rectangle` function. The rectangle color is set to blue `(255, 0, 0)` and the thickness is set to 2.

6. Exporting the Result: The modified image with rectangles drawn around the detected faces is saved as `face_detected.png` using `cv2.imwrite`.

7. Print Confirmation: A message is printed to indicate that the photo with detected faces has been successfully exported.

Remember to replace the input image name (`test3.jpg`) with the actual image you want to perform face detection on. Also, make sure you have the `haarcascade_frontalface.xml` classifier file available in the same directory. This classifier is based on Haar cascades and is a pre-trained model for face detection provided by OpenCV.

Here test3.jpg is the input data.

![test3](Old-Version/test3.jpg)

The Haarcascade processed output:

![output](Old Version/face_detected (1).png)

## About Haarcascade Frontalface

`haarcascade_frontalface` is a pre-trained Haar Cascade classifier provided by OpenCV, a popular computer vision library. Haar Cascade classifiers are machine learning-based object detection methods that are used to detect objects (in this case, faces) within images or video frames. These classifiers work by analyzing patterns of intensity gradients in the image.

The `haarcascade_frontalface.xml` file is a configuration file that contains the information needed to perform frontal face detection. It has been trained on a large number of positive and negative images to recognize specific patterns associated with human faces, particularly when they are facing the camera directly (frontal faces).

When you use the `CascadeClassifier` in OpenCV and load the `haarcascade_frontalface.xml` classifier, you're essentially utilizing a model that has already learned how to detect frontal faces. The classifier works by sliding a window of different sizes across the input image and looking for patterns that match the characteristics of faces. If the classifier detects a pattern that resembles a face, it marks that region as a potential face detection.

Keep in mind that while Haar Cascade classifiers are fast and lightweight, they might not perform well under various conditions, such as when faces are tilted, occluded, or captured under different lighting conditions. For more advanced scenarios, deep learning-based object detection methods like Single Shot MultiBox Detector (SSD) or You Only Look Once (YOLO) are often used, as they can handle more complex cases effectively.


[Old Google Colab File](https://colab.research.google.com/drive/1xGTUCZYVUExRcsMH_h-o8XjAPBpamNRd?usp=sharing)
