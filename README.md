# 😊 Facial Expression Classification using CNN

This project is a Convolutional Neural Network (CNN)-based model designed to classify human facial expressions into different emotion categories using image data.

---

## 📌 Features

- Facial image preprocessing (grayscale, resizing, normalization)
- CNN architecture built using TensorFlow/Keras
- Trained on facial expression datasets (e.g., FER2013, CK+, or custom)
- Classifies emotions like happy, sad, angry, surprise, neutral, etc.
- Includes visualization of training results and confusion matrix

---

## 📁 Dataset

- Input: 48x48 grayscale images of faces  
- Emotion labels: Happy, Sad, Angry, Fear, Surprise, Neutral, etc.  
- Dataset used: FER2013 or other public facial expression datasets

---

## 🧠 Model Architecture

```python
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
