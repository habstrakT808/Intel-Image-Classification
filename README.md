# 🏞️ Intel Image Classification Project

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> A deep learning project for classifying natural scene images using Convolutional Neural Networks.
  ![Image](https://github.com/user-attachments/assets/0f15aeac-edb7-46a4-9832-97658b9b8cbd)
<p align="center">
  
</p>

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Setup](#-setup)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Training Process](#-training-process)
- [Results](#-results)
- [Model Export](#-model-export)
- [Inference](#-inference)
- [Contributing](#-contributing)
- [License](#-license)

## 🔍 Overview

This project implements a deep learning solution for the Intel Image Classification dataset challenge. It classifies natural scenes into 6 categories: buildings, forest, glacier, mountain, sea, and street. The implementation uses TensorFlow/Keras to build, train and evaluate a Convolutional Neural Network (CNN) model.

The project follows a complete machine learning pipeline:
1. Dataset preparation and exploration
2. Data preprocessing and augmentation
3. Model architecture design
4. Training with callbacks
5. Evaluation and visualization
6. Model export in multiple formats
7. Inference testing

## 📊 Dataset

The Intel Image Classification dataset contains around 25,000 images of size 150x150 distributed under 6 categories:
- 🏢 Buildings
- 🌳 Forest
- ❄️ Glacier
- ⛰️ Mountain
- 🌊 Sea
- 🛣️ Street

![Image](https://github.com/user-attachments/assets/3e205575-16fd-4104-a2a7-5011d8e075e9)

## 🚀 Setup

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- Required libraries (listed in requirements.txt)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/intel-image-classification.git
cd intel-image-classification
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Intel Image Classification dataset and extract it in the project directory.

## 📂 Project Structure

```
intel-image-classification/
│
├── intel_image_dataset/           # The dataset directory
│   ├── seg_train/                 # Original training set
│   ├── seg_test/                  # Original test set
│   └── seg_pred/                  # Original prediction set
│
├── intel_dataset_split/           # Custom dataset split
│   ├── training/                  # Custom training set
│   ├── validation/                # Custom validation set
│   └── testing/                   # Custom testing set
│
├── logs/                          # TensorBoard logs
├── saved_checkpoints/             # Model checkpoints
├── saved_model/                   # SavedModel format
├── tflite/                        # TFLite model
├── tfjs_model/                    # TensorFlow.js model
├── model_export/                  # Other exported models
│
├── main.py                        # Main project script
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## 🧠 Model Architecture

This project implements a custom CNN architecture optimized for scene classification:

```python
model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
    BatchNormalization(),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Second Convolutional Block
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Third Convolutional Block
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Fourth Convolutional Block
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Flatten layer
    Flatten(),

    # Dense layers
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

Key features of the architecture:
- Four convolutional blocks with increasing filter sizes (32→64→128→256)
- Batch normalization for stabilizing learning
- Dropout for regularization
- Dense output layer with softmax activation

## 🏋️ Training Process

The model is trained with:
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Categorical Cross-Entropy
- **Metrics**: Accuracy
- **Batch Size**: 32
- **Image Size**: 150×150 pixels
- **Epochs**: Maximum 50 (with early stopping)

### Data Augmentation

To improve model generalization, the following augmentations are applied:
- Rotation (up to 20 degrees)
- Width/height shifts (up to 20%)
- Shear transformations (up to 20%)
- Zoom (up to 20%)
- Horizontal flips

### Callbacks

Training utilizes several callbacks:
- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Model Checkpoint**: Saves the best model based on validation accuracy
- **ReduceLROnPlateau**: Reduces learning rate when validation loss plateaus
- **TensorBoard**: Logs metrics for visualization
- **Target Accuracy Callback**: Stops training when accuracy reaches 95%

![Image](https://github.com/user-attachments/assets/a28556c7-69d4-45cf-b028-3cd1147201b0)

## 📈 Results

The model achieves excellent performance on the classification task:

| Metric | Training | Validation | Testing |
|--------|----------|------------|---------|
| Accuracy | ~98% | ~92% | ~91% |
| Loss | ~0.08 | ~0.25 | ~0.27 |

### Confusion Matrix

![Image](https://github.com/user-attachments/assets/26cb658c-6857-47ea-a7c2-fe25e6502203)

The confusion matrix shows that the model performs well across all classes, with minimal confusion between similar categories like "glacier" and "mountain".

## 📦 Model Export

The model is exported in multiple formats for different deployment scenarios:

1. **SavedModel Format**: Standard TensorFlow format
2. **.keras/.h5**: Standard Keras format
3. **TF-Lite**: Optimized for mobile and edge devices
4. **TensorFlow.js**: For web browser deployment

## 🔮 Inference

The project includes functionality to test the trained model on sample images:

```python
# Example code for inference
def preprocess_image_for_prediction(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Load and preprocess image
test_img = preprocess_image_for_prediction('path/to/image.jpg')

# Make prediction
predictions = scene_classifier.predict(test_img)
predicted_class = np.argmax(predictions[0])
predicted_class_name = class_names[predicted_class]
confidence = predictions[0][predicted_class]

print(f"Predicted category: {predicted_class_name}")
print(f"Confidence: {confidence:.4f}")
```

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<p align="center">
  Made with ❤️ by <a href="[(https://github.com/habstrakT808)">Hafiyan Al Muqaffi Umary</a>
</p>
