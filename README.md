
# Audio Classification

## Overview

This project focuses on classifying audio into three categories: Alarm, Water, and Other. We started with unbalanced data but addressed this by collecting more samples, ensuring better accuracy in the final model.

## Data Collection and Organization

Initially, the data we collected was unbalanced, affecting the model's accuracy. To fix this, we gathered additional audio samples of water and other objects to balance the dataset.

### Data Organization

The collected audio files were scattered across different folders. To streamline the process, we created a script, `data_arranger.py`, that:

- Organizes the audio files into three categories: Alarm, Water, and Other.
- Splits the data into 95% for training and 5% for testing.
- Prepares a sample directory with one test file for each category (Alarm, Water, Other) to use for inference.

### Data Preprocessing

To make the audio data ready for the model, we used the `data_preprocessor.py` script, which:

- Extracts features from the audio using Short-Time Fourier Transform (STFT) and Mel-Frequency Cepstral Coefficients (MFCC).
- Splits the training data into 85% for training and 15% for validation.
- Converts the labels into a format suitable for the model.
- Standardizes the features using a standard scaler from `sklearn`.
- Applies data augmentation to enhance the dataset.

### Accessing the Data

The processed data is organized into three main directories:

- **Raw**: Contains the original audio files.
- **Balanced**: Organized into Train, Test, and Inference sets, with subfolders for Alarm, Water, and Other.
- **Features**: Contains the extracted features from the preprocessing step.

You can access the data here: [Download Data](https://drive.google.com/open?id=1yNeQqvHEEPm8eN5ullnM34XkU6KoV03M&usp=drive_fs).

## Model Training

With the data preprocessed, the training process was straightforward. Using the `train.py` script:

- The data was loaded and further balanced using the SMOTE algorithm.
- We trained the model for 30 epochs with a batch size of 32, using the training and validation sets.
- The model structure is similar to AlexNet but optimized for 1D data, using two 1D convolutional layers followed by a 1D MaxPooling layer.
- After training, the model achieved 93% accuracy, successfully classifying all three categories.

You can download the trained model weights here: [Download Weights](https://drive.google.com/open?id=1V6ES-tPA48wDQueZg_MxMXmIZMX2CbBY&usp=drive_fs).

## Model Testing

For deployment, we needed to convert the model to a format suitable for embedded systems. The `test.py` script:

- Converts the trained model to TensorFlow Lite (TFLite) format.
- Runs inference using the TFLite model and compares its performance with the original Keras model.
