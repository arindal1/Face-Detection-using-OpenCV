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

[Face Detection - Google Colab File](https://colab.research.google.com/drive/1xGTUCZYVUExRcsMH_h-o8XjAPBpamNRd?usp=sharing)
