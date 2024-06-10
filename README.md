# Audio classifier trained on FSD50k dataset

This is an attempt to train a classifier for our use case on the [FSD50k
dataset][1]. The [AudioSet][2] dataset is vast since clips are collected from
[YouTube][3] but that means that many clips are in poor quality and contain
multiple sound sources. Additionally, there could be legal implications due to
YouTube's Terms of Service.

## Audio preprocessing

Audio classification is typically done with [Convolutional Nerual Networks
(CNN)][9] which are also used for image recognition. Additionally, audio data in
an audio classification task typically converted into a [Mel Spectrogram][10]
before it is provided to the classifier (this is called preprocessing).

The following links explain core concepts of audio preprocessing required to
build audio classifiers:
1. [Short-Time Fourier Transform (STFT)][4]
2. [STFT and Spectrograms in Python][5]
3. [Mel Spectrograms][6]
4. [Mel Spectrograms in Python][7]

**NOTE:** Current classifier is a carbon copy of the classifier from [this
article][9].

[1]:  https://zenodo.org/records/4060432
[2]:  https://research.google.com/audioset/dataset/index.html
[3]:  https://www.youtube.com/
[4]:  https://www.youtube.com/watch?v=-Yxj3yfvY-4
[5]:  https://www.youtube.com/watch?v=3gzI4Z2OFgY
[6]:  https://www.youtube.com/watch?v=9GHCiiDLHQ4
[7]:  https://www.youtube.com/watch?v=TdnVE5m3o_0
[8]:  https://towardsdatascience.com/sound-event-classification-using-machine-learning-8768092beafc
[9]:  https://en.wikipedia.org/wiki/Convolutional_neural_network
[10]: https://ketanhdoshi.github.io/Audio-Mel/
