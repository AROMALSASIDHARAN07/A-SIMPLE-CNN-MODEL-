# Intel Image Classification using CNN

## 📌 Project Overview
This project implements a **Convolutional Neural Network (CNN)** to classify natural scene images into six categories using the **Intel Image Classification Dataset**.  
The model is built, trained, and evaluated using **TensorFlow/Keras**, with proper preprocessing and performance evaluation metrics.

---

## 🗂 Dataset Description
The Intel Image Classification dataset contains images grouped into the following folders:

- **seg_train** – Training dataset (labeled)
- **seg_test** – Test dataset (labeled)
- **seg_pred** – Prediction dataset (unlabeled, used for inference)

### Classes:
- Buildings  
- Forest  
- Glacier  
- Mountain  
- Sea  
- Street  

---

## ⚙️ Data Preprocessing
- Images are initially loaded at **256 × 256**
- Resized to **150 × 150**
- Pixel values normalized to **[0, 1]**
- Data loaded using `image_dataset_from_directory`

---

## 🧠 Model Architecture
The CNN model consists of:
- Convolutional layers with ReLU activation
- Max Pooling layers for spatial reduction
- Fully connected (Dense) layers
- Dropout for regularization
- Softmax output layer for multi-class classification

---

## 🚀 Model Training
- Optimizer: **Adam**
- Loss Function: **Sparse Categorical Crossentropy**
- Validation performed using the test dataset
- **Early Stopping** used to prevent overfitting

```python
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=50,
    callbacks=[early_stop],
    verbose=1
)
