import os
import tensorflow as tf
import librosa
import pandas
import random
import numpy as np

import augmentation

def load_wav_16k_mono(filename: str, duration=None) -> np.ndarray:
    wav, _ = librosa.load(filename, sr=16000, mono=True, duration=duration)
    return wav

def get_log_mel_spectrogram(wav: np.ndarray) -> np.ndarray:
    mel_spectrogram = librosa.feature.melspectrogram(y=wav, sr=16000, n_fft=512,
                                                     win_length=512, hop_length=512,
                                                     n_mels=60, fmin=0, fmax=8000,
                                                     center=False)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, top_db=80.0)
    return log_mel_spectrogram

def preprocess(wav: np.ndarray) -> np.ndarray:
    assert len(np.shape(wav)) == 1, f"sound of shape {np.shape(wav)} not one dimensional"
    NUM_SAMPLES = 47616 # Multiple of hop length so that mel-spectrogram conversion does not pad the output.
    wav         = wav[:NUM_SAMPLES]
    pad_size    = NUM_SAMPLES - np.shape(wav)[0]
    wav         = np.pad(wav, (pad_size, 0), 'constant', constant_values=(0, 0))
    return wav

def get_class_weights(dataframe: pandas.DataFrame, class_to_id_map: dict[str, int]):
    n_total = len(dataframe)
    calculate_weight = lambda n_positive: n_total / (n_positive * 2.0)
    value_counts = dataframe['class_id'].value_counts()
    return {v: calculate_weight(value_counts.iat[v]) for _, v in class_to_id_map.items()}

def augment_alarms(original_dataset: pandas.DataFrame):
    noises = original_dataset[original_dataset["class_id"] == 0]
    alarms = original_dataset[original_dataset["class_id"] == 1]
    augmenters = [
        augmentation.create_noise_augmenter(np.stack(noises['samples'].to_numpy()), 0.8, 1.2),
        augmentation.create_white_noise_augmenter(0.8, 1.2),
        augmentation.create_time_stretch_augmenter(0.8, 1.2),
        augmentation.create_pitch_shift_augmenter(16000, 0, 5),
        augmentation.create_invert_polarity_augmenter(),
        augmentation.create_gain_augmenter(0.5, 1.5),
        augmentation.create_rotate_augmenter(0, 8000),
    ]
    apply_probability = 2 * 1.0 / len(augmenters)
    def augment_sample(sample: np.ndarray):
        for augmenter in augmenters:
            if random.random() < apply_probability:
                sample = augmenter(sample)
        return sample

    augmented_alarms = pandas.concat([alarms] * 140, ignore_index=True)
    augmented_alarms['samples'] = augmented_alarms['samples'].map(augment_sample)
    return pandas.concat([original_dataset, augmented_alarms], ignore_index=True)

def create_dataset(dataset_dir:     str,
                   class_to_id_map: dict[str, int],
                   batch_size:      int) -> tuple[tf.data.Dataset, tf.data.Dataset, dict[str, float]]:
    filename_label_list = np.asarray([(os.path.join(dataset_dir, label, file), id)
         for label, id in class_to_id_map.items()
         for file in os.listdir(os.path.join(dataset_dir, label))])

    dataframe             = pandas.DataFrame(filename_label_list, columns=['filename', 'class_id'])
    dataframe['class_id'] = pandas.to_numeric(dataframe['class_id'])
    dataframe['samples']  = dataframe['filename'].map(lambda filename: preprocess(load_wav_16k_mono(filename)))

    print('Augmenting the dataset...', end='', flush=True)
    dataframe = augment_alarms(dataframe)
    print('Done!')

    dataframe['spectrogram'] = dataframe['samples'].map(get_log_mel_spectrogram)
    dataframe['spectrogram'] = dataframe['spectrogram'].map(lambda spectrogram: spectrogram.astype(np.float32))
    spectrograms_shape       = (len(dataframe['spectrogram']), ) + np.shape(dataframe['spectrogram'][0])
    dataset = tf.data.Dataset.from_tensor_slices(
            (np.concatenate(dataframe['spectrogram'].to_numpy()).reshape(spectrograms_shape),
             dataframe['class_id'].to_numpy(dtype=np.float32)))

    print('Shuffling the dataset...', end='', flush=True)
    dataset = dataset.cache().shuffle(buffer_size=dataset.cardinality()).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    print('Done!')

    train_dataset_len  = int(0.8 * len(dataset))
    train_dataset      = dataset.take(train_dataset_len)
    validation_dataset = dataset.skip(train_dataset_len)
    return train_dataset, validation_dataset, get_class_weights(dataframe, class_to_id_map)

def plot_dataset():
    import matplotlib.pyplot as plt
    DATASET_DIR     = '/home/rudolf/dev/nomo/dataset/alarmset-custom'
    CLASS_TO_ID_MAP = {'other': 0, 'alarm': 1}
    BATCH_SIZE      = 32

    train_dataset, _, _    = create_dataset(DATASET_DIR, CLASS_TO_ID_MAP, BATCH_SIZE)
    train_dataset_as_list  = list(train_dataset.as_numpy_iterator())
    batch_x, batch_y       = train_dataset_as_list[2]
    
    _, axis = plt.subplots(4, 8)
    for i in range(4):
        for j in range(8):
            s = i * 8 + j 
            title = 'other' if batch_y[s] == 0.0 else 'alarm'
            librosa.display.specshow(batch_x[s], ax=axis[i, j]) #y_axis='mel', x_axis='time',
            axis[i, j].set_title(title)
    plt.show()
