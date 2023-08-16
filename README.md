# Face Detection using OpenCV
*This is a personal project, a Face Detection algorithm in Python using OpenCV*

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

### [Modern Face Recognition with Deep Learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)
This an article by [Adam Geitgey](https://medium.com/@ageitgey?source=post_page-----c3cffc121d78--------------------------------) about how Modern Face Recognition works, in both theoritical and practical ways. It's a 13-15 minutes read. You can read this article before starting this project yourself, it's very informative and helpful!


# Real-time Face Recognition using OpenCV and simple_facerec

This project demonstrates real-time face recognition using the OpenCV library along with the `simple_facerec` library. The program captures video frames from your webcam and detects known faces in the frames by comparing them with previously encoded faces.

## Getting Started

Follow the steps below to set up and run the project on your local machine.

### Prerequisites

- Python 3.x
- OpenCV (install with `pip install opencv-python`)
- `simple_facerec` (keep the file in the base directory)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/arindal1/Face-Detection-using-OpenCV.git
   cd face-recognition-project
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Place face images for encoding in the `InData/` directory.
2. Run the script:
   ```bash
   python main.py
   ```
3. The script will open a window showing the webcam feed with recognized faces highlighted.

Press the "Esc" key to exit the application.

## How it Works

1. The program initializes the `simple_facerec` library and loads face encodings from the `InData/` directory.
2. It captures video frames from the webcam using OpenCV.
3. Detected faces in each frame are compared with the loaded encodings to recognize known faces.
4. Recognized faces are highlighted with bounding boxes and labels on the webcam feed.

## Breakdown of the Scripts

### Base File
The `base.py` is the base file. A Python script using the OpenCV and face_recognition libraries to perform face recognition on two images. The script first loads and encodes faces from the provided images, and then compares the facial encodings to determine if the faces in the images match. Finally, it displays the images using OpenCV's `imshow` function.

Here's a breakdown of the script:

1. Import the necessary libraries:
   ```python
   import cv2
   import face_recognition
   ```

2. Load the first image (`elon1.jpg`) and convert it to RGB color format:
   ```python
   img = cv2.imread("InData/elon1.jpg")
   rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   ```

3. Encode the face(s) present in the first image:
   ```python
   img_encoding = face_recognition.face_encodings(rgb_img)[0]
   ```

4. Load the second image (`musk1.jpg`) and convert it to RGB color format:
   ```python
   img2 = cv2.imread("TestData/musk1.jpg")
   rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
   ```

5. Encode the face(s) present in the second image:
   ```python
   img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]
   ```

6. Compare the two face encodings to determine if they match:
   ```python
   result = face_recognition.compare_faces([img_encoding], img_encoding2)
   ```

7. Print the result of the face comparison:
   ```python
   print("Result: ", result)
   ```

8. Display the two images using OpenCV's `imshow` function:
   ```python
   cv2.imshow("Img", img)
   cv2.imshow("Img 2", img2)
   cv2.waitKey(0)
   ```

If the result is `[True]`, it means that the faces in the two images match; if it's `[False]`, the faces don't match.

Make sure that you have the required image files (`elon1.jpg` and `musk1.jpg`) in the specified paths ("InData/" and "TestData/") before running the script. Also, ensure you have the OpenCV and face_recognition libraries installed in your environment.

### Real Time Detection
The `facedetector.py` is the file for real time face detection, a Python script that uses the `simple_facerec` library along with OpenCV to perform real-time face recognition using your webcam feed. The script first loads face encodings from a folder, then utilizes the webcam to detect and recognize faces in the live video stream.

Here's a breakdown of the script:

1. Import the necessary libraries:
   ```python
   import cv2
   from simple_facerec import SimpleFacerec
   ```

2. Initialize the `SimpleFacerec` object and load encoded images from the specified folder ("InData/"):
   ```python
   sfr = SimpleFacerec()
   sfr.load_encoding_images("InData/")
   ```

3. Load the webcam capture device:
   ```python
   cap = cv2.VideoCapture(2)
   ```

4. Start an infinite loop to continuously process frames from the webcam:
   ```python
   while True:
       ret, frame = cap.read()
   ```

5. Detect known faces in the current frame using the loaded encodings:
   ```python
   face_locations, face_names = sfr.detect_known_faces(frame)
   ```

6. Iterate over the detected face locations and corresponding names:
   ```python
   for face_loc, name in zip(face_locations, face_names):
       y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

       # Draw the name and bounding box around the detected face
       cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
       cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
   ```

7. Display the processed frame in a window named "Frame":
   ```python
   cv2.imshow("Frame", frame)
   ```

8. Check for a key press and break the loop if the "Esc" key (key code 27) is pressed:
   ```python
   key = cv2.waitKey(1)
   if key == 27:
       break
   ```

9. Release the webcam capture device and close all OpenCV windows when the loop is exited:
   ```python
   cap.release()
   cv2.destroyAllWindows()
   ```

This script essentially captures video frames from your webcam, processes them to detect and recognize faces using the loaded encodings, and displays the frames with recognized names and bounding boxes. Pressing the "Esc" key will stop the script and close the OpenCV windows. Make sure you have the `simple_facerec` library installed in your environment, and ensure your webcam is accessible and properly configured.

## Accuracy

Calculating the accuracy of a face recognition model like the one you've implemented can be a bit more involved than a traditional machine learning classification model because it doesn't follow the typical training/testing split. However, you can still evaluate the performance of your face recognition system by considering some aspects:

1. **Known Face Identification Accuracy**: Since your model is detecting known faces from a set of loaded encodings, you could evaluate how well it correctly identifies the pre-registered faces in real-time. This could be done by comparing the recognized names with the actual names of the individuals in the frame and calculating the percentage of correct identifications.

2. **False Positives and False Negatives**: You can also keep track of false positives (recognizing a face that shouldn't be recognized) and false negatives (not recognizing a face that should be recognized). This will provide insight into the model's performance with respect to both over-recognition and under-recognition.

Remember that these calculations are rough estimates and may not provide a complete picture of the model's performance, especially considering factors like varying lighting conditions, angles, and other real-world challenges. Also, the effectiveness of the accuracy calculation largely depends on the quality of the encodings, the diversity of the training data, and the robustness of the recognition process.

## Contributing

Contributions to this project are welcome! If you have any suggestions, improvements, or bug fixes, feel free to open an issue or submit a pull request.

## Acknowledgements

- This project uses the [simple_facerec](https://github.com/davisking/dlib) library for face recognition.
- Special thanks to the authors and contributors of the OpenCV and `simple_facerec` libraries for their valuable work.

## Contact

If you have any questions or want to connect, feel free to reach out:

- GitHub: [arindal1](https://github.com/arindal1)
- LinkedIn: [Arindal](https://www.linkedin.com/in/arindalchar/)

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

![test3](https://github.com/arindal1/Face-Detection-using-OpenCV/blob/main/Old%20Version/test3.jpg)

The Haarcascade processed output:

![output](https://github.com/arindal1/Face-Detection-using-OpenCV/blob/main/Old%20Version/face_detected%20(1).png)

## About Haarcascade Frontalface

`haarcascade_frontalface` is a pre-trained Haar Cascade classifier provided by OpenCV, a popular computer vision library. Haar Cascade classifiers are machine learning-based object detection methods that are used to detect objects (in this case, faces) within images or video frames. These classifiers work by analyzing patterns of intensity gradients in the image.

The `haarcascade_frontalface.xml` file is a configuration file that contains the information needed to perform frontal face detection. It has been trained on a large number of positive and negative images to recognize specific patterns associated with human faces, particularly when they are facing the camera directly (frontal faces).

When you use the `CascadeClassifier` in OpenCV and load the `haarcascade_frontalface.xml` classifier, you're essentially utilizing a model that has already learned how to detect frontal faces. The classifier works by sliding a window of different sizes across the input image and looking for patterns that match the characteristics of faces. If the classifier detects a pattern that resembles a face, it marks that region as a potential face detection.

Keep in mind that while Haar Cascade classifiers are fast and lightweight, they might not perform well under various conditions, such as when faces are tilted, occluded, or captured under different lighting conditions. For more advanced scenarios, deep learning-based object detection methods like Single Shot MultiBox Detector (SSD) or You Only Look Once (YOLO) are often used, as they can handle more complex cases effectively.


[Old Google Colab File](https://colab.research.google.com/drive/1xGTUCZYVUExRcsMH_h-o8XjAPBpamNRd?usp=sharing)
