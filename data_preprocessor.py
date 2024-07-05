import librosa
import os
import random
import numpy as np
import noisereduce as nr
# from noisereduce.noisereducev1 import reduce_noise
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

def get_STFT(file):
    # read audio data
    audio_data, sample_rate = librosa.load(file, sr=16000)

    # noise reduction
    noisy_part = audio_data[0:25000]
    #     reduced_noise = nr.reduce_noise(audio_clip=audio_data, noise_clip=noisy_part, verbose=False)
    # reduced_noise = nr.reduce_noise(y=audio_data, y_noise=noisy_part, sr=sample_rate)
    reduced_noise = audio_data

    # trimming
    trimmed, index = librosa.effects.trim(reduced_noise, top_db=20, frame_length=512, hop_length=64)

    # extract features
    stft = np.abs(librosa.stft(trimmed, n_fft=512, hop_length=256, win_length=512))

    return stft

def extract_features(trimmed):
    mfccs = librosa.feature.mfcc(y=trimmed, sr=16000, n_mfcc=13)  # Compute MFCCs
    mfccs_mean = np.mean(mfccs.T, axis=0)  # Compute the mean of the MFCCs
    return mfccs_mean.flatten()

def feature_extraction(data_dir, name):
    # Load images and labels
    audios = []
    labels = []
    class_names = []

    # Get the list of class directories
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    # Iterate through each class directory
    for i, class_dir in enumerate(class_dirs):
        class_names.append(class_dir)
        class_audio = os.listdir(os.path.join(data_dir, class_dir))

        # Iterate through each image in the class directory
        for audio_file in tqdm(class_audio, total=len(class_audio), desc=f'Extracting {name} {class_dir}'):
            audio_path = os.path.join(data_dir, class_dir, audio_file)

            trimmed = get_STFT(audio_path)
            audio_array = extract_features(trimmed)

            audios.append(audio_array)
            labels.append(i)

    # Convert lists to numpy arrays
    X = np.array(audios)
    y = np.array(labels)

    return X, y, class_names

def augment_data(data):
    augmented_data = []
    for sample in data:
        if random.random() > 0.5:
            sample = sample + random.uniform(-0.1, 0.1)  # Random noise
        if random.random() > 0.5:
            sample = sample * random.uniform(0.8, 1.2)  # Random scaling
        augmented_data.append(sample)
    return np.array(augmented_data)

if __name__ == "__main__":

    X_train, y_train, class_names = feature_extraction('data/balanced/train', 'train')

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.10, stratify=y_train)

    X_test, y_test, _ = feature_extraction('data/balanced/test', 'test')

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = augment_data(X_train)

    featuresPath = "data/features/"

    os.makedirs(featuresPath, exist_ok=True)

    np.save(os.path.join(featuresPath, 'class_names.npy'), class_names)

    np.save(os.path.join(featuresPath, 'X_train.npy'), X_train)
    np.save(os.path.join(featuresPath, 'y_train.npy'), y_train)

    np.save(os.path.join(featuresPath, 'X_val.npy'), X_val)
    np.save(os.path.join(featuresPath, 'y_val.npy'), y_val)

    np.save(os.path.join(featuresPath, 'X_test.npy'), X_test)
    np.save(os.path.join(featuresPath, 'y_test.npy'), y_test)
