## AI-Powered Plant Disease Detection

A complete, end-to-end Python script to build and train a Convolutional Neural Network (CNN) for identifying 38 plant diseases from leaf images. Optimized to run efficiently within the standard 12GB RAM of Google Colab, the solution automates data download, preprocessing, model construction, training, evaluation, and interactive prediction for user-uploaded images.

### Project Objective
Develop a robust, computationally efficient deep learning model for accurate plant disease recognition, accessible on free cloud platforms like Google Colab—eliminating the need for high-end hardware.

### Memory-Efficient Design
- **Strategic Data Slicing:** Trains on a key subset of PlantVillage, reducing RAM use while preserving diverse learning signals.
- **Efficient tf.data Pipeline:** Utilizes TensorFlow Datasets (`tfds`) to stream images from disk, excluding memory-heavy `.cache()`.
- **Balanced CNN Architecture:** Carefully tuned layers and filter counts optimize performance vs. resource consumption.

### Key Features
- **Colab-Ready Efficiency:** Runs smoothly within 12GB RAM.
- **Strong Accuracy:** Delivers high validation accuracy (91%).
- **Fast Training:** Automatic GPU check & practices cut training time.
- **End-to-End Automation:** One script for all steps—data, model, train, evaluate, save.
- **Interactive Prediction:** Simple module lets users upload images and get instant results.

### Technology Stack
- **Frameworks/Libraries:** Python, TensorFlow, Keras, TFDS, NumPy
- **Tools:** Google Colab, Matplotlib

---
