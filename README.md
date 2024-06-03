# Audio-classifier
## Overview
This project can be used to build ML model for audio sound classification, for example: water sounding detector or alarm sounding detector.

Let’s come to the sound classification part. Let’s say the problem is to classify a given audio clip as one of the following events using only sound.
- Water — water sounding like washing hands with a sink, water tap and shower, etc.
- Alarm - Several alarm sounding like warning, alert, error and so on.
- Other — None of the above events.

Audio dataset which used for training model can be downloaded from the following link: [here](https://drive.google.com/file/d/1s0EdfXSyJAHA8hfL8d3k-SLCVCuLg5Dp/view?usp=sharing)

The sound event classification is done by performing Audio preprocessing, Feature extraction and classification.
### Audio preprocessing
First, we need to come up with a method to represent audio clips (.wav files). Then, the audio data should be preprocessed to use as inputs to the machine learning algorithms. The Librosa library provides some useful functionalities for processing audio with python. The audio files are loaded into a numpy array using Librosa. The array will consist of the amplitudes of the respective audio clip at a rate called ‘Sampling rate’. (The sampling rate usually would 16000, 22050 or 44100)

Having loaded an audio clip into an array, the first challenge, noisiness should be addressed. The Audacity uses ‘spectral gating’ algorithm to suppress the noise in the audio.
The code lines below shows how to implment it.
```python
    import noisereduce as nr
    import librosa
    import numpy as np
    
    audio_data, sample_rate = librosa.load(file, sr=16000)
    print(f"file={file} sample rate={sample_rate} audio_data={audio_data.shape}")
    # noise reduction
    noisy_part = audio_data[0:25000]
    reduced_noise = nr.reduce_noise(audio_clip=audio_data, noise_clip=noisy_part, verbose=False)
    # Visualize
    print("Original audio file:")
    plotAudio(audio_data)
    print("Noise removed audio file:")
    plotAudio(reduced_noise)

```
The result is as follows:
<p align="center">
    <img alt="" src="https://github.com/watch-nomo/audio-classifier/assets/93747826/5972a268-deee-490b-bb2a-fec00d8a406a" width=500/>
</p>
The resulting audio clip is containing an empty length (unnecessary silence) in it. Let’s trim the leading and trailing parts which are silence than a threshold loudness level.

```python
    trimmed, index = librosa.effects.trim(reduced_noise, top_db=20, frame_length=512, hop_length=64)
    print(“Trimmed audio file:”)
    plotAudio(trimmed)
```
The result is as follows:
<p align="center">
    <img alt="" src="https://github.com/watch-nomo/audio-classifier/assets/93747826/e90c456c-376b-4148-bf1f-b501bb46db93" width=500/> 
</p>

### Feature extraction
The preprocessed audio files themselves cannot be used to classify as sound events. We have to extract feature from the audio clips to make the classification process more efficient and accurate. Let’s extract the absolute values of Short-Time Fourier Transform (STFT) from each audio clip. To calculate STFT, Fast Fourier transform window size(n_fft) is used as 512. According to the equation n_stft = n_fft/2 + 1, 257 frequency bins(n_stft) are calculated over a window size of 512. The window is moved by a hop length of 256 to have a better overlapping of the windows in calculating the STFT.
```python
stft = np.abs(librosa.stft(trimmed, n_fft=512, hop_length=256, win_length=512))
```
The number of frequency bins, window length and hop length are determined empirically for the dataset. There is no universal set of values for the parameters in the feature generation. This will be discussed again in the Tuning and enhancing Results section.

Let’s review the meaning of the absolute STFT features of an audio clip. Consider an audio clip which has a `t` number of samples in it. Say we are obtaining an `f` number of frequency bins in the STFT. Consider the window length is `w` and window’s hop length `h`. When calculating STFT, a series of windows are obtained sliding a fixed `w` length window by a step of `h`. This will produce `1+(t-w)/h` number of windows. For each such window, the amplitudes of frequency bins (in Hz) in the range `0` to `sampling_rate/2` are recorded. The frequency range is equally divided when determining the values of the frequency bins. For example, consider STFT bins with `n_fft=16` and sampling rate of `22050`. Then there will be `9` frequency bins having the following values(in Hz).

`[     0   ,   1378.125,   2756.25 ,   4134.375, 5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025   ]`

The absolute STFT features of an audio clip is a 2-dimensional array which contains mentioned frequency amplitude bins for each window.
Since sound events have different durations(number of samples), the 2-d feature arrays are flattened using mean on the frequency axis. Thus, the audio clips will be represented using an array of fixed size `257` (number of STFT frequency bins). This seems like a bad representation of the audio clip since it does not contain temporal information. But every given audio event has its unique frequency range. For example, the scratching sound of the sweeping-event makes its feature vector having more high-frequency amplitudes than Falling-event. Finally, the feature vector is normalized by min-max normalization. Normalization is applied to make every sound event lie on a common loudness level. (As introduced, the amplitude is an audio property and we should not use amplitude in this use case to differentiate sound events). The following figure shows the Normalized STFT Frequency signatures of sound events captured by the absolute STFT features.
<p align="center">
    <img alt="" src="https://github.com/watch-nomo/audio-classifier/assets/93747826/5128ef0a-7221-4db3-8672-19bdc3f2faa8" width=500/>
</p>

### Event classification
Now, the sound events are preprocessed and represented efficiently using STFT features. The STFT features are used to train a fully connected Neural Network(NN) which will be used to classify new sound events. The NN consists of `5` fully connected layers. The layers are having `256`, `256`, `128`, `128` and `8` neurons in them. All layers are having ReLU activation function and the 4ᵗʰ layer is having a dropout layer to reduce the overfitting to the training data. The neural network (model) can be easily built using Keras Dense layers. The model is compiled using the ‘Adam’ optimizer.
```python
# build model
model = Sequential()model.add(Dense(256, input_shape=(257,)))
model.add(Activation(‘relu’))

model.add(Dense(256))
model.add(Activation(‘relu’))

model.add(Dense(128))
model.add(Activation(‘relu’))

model.add(Dense(128))
model.add(Activation(‘relu’))
model.add(Dropout(0.5))
model.add(Dense(num_labels))
model.add(Activation(‘relu’))
model.compile(loss=’categorical_crossentropy’, metrics=[‘accuracy’], optimizer=’adam’)
```
The above NN model classifies unseen audio events. The following figure shows the normalized confusing matrix corresponding to a prediction.
<p align="center">
    <img alt="" src="https://github.com/watch-nomo/audio-classifier/assets/93747826/fc2d1e2e-a52e-46fe-b81e-c74f8f6b2ffc" width=500/>
</p>

## How to train model
First, please download dataset from the following link: [here](https://drive.google.com/file/d/1s0EdfXSyJAHA8hfL8d3k-SLCVCuLg5Dp/view?usp=sharing) and then unzip file to place the project path

And then, please run the following scripts in turn
```bash
python make_dataset_16k.py
python train_16k.py
```
## How to convert model to tflite and how to compare Keras model to quantized tflite
To do it, please run the following scripts to convert keras model into tflite
```bash
python model_conv.py
```
