import glob
import os
import numpy as np
import librosa
import sys
import tensorflow as tf
from keras.models import load_model

target_length = 3341 
# This is model input shape, to refer this number, use the following CLI: print(load_model(keras_model_file).input)
def pad_features(features, target_length):
    if len(features) < target_length:
        padded_features = np.zeros(target_length)
        padded_features[:len(features)] = features
    else:
        padded_features = features[:target_length]
    return padded_features

def get_STFT(file, trim_flag=False, same_training=False):
    audio_data, sample_rate = librosa.load(file, sr=16000)

    zero_data = np.zeros((160,))
    audio_data = zero_data.tolist() + audio_data.tolist()
    audio_data = np.array(audio_data)
    audio_data[480:512] = 0
    reduced_noise = audio_data

    trimmed = reduced_noise
    if trim_flag or same_training:
        trimmed, index = librosa.effects.trim(reduced_noise, top_db=20, frame_length=512, hop_length=64)
    # extract features
    if same_training:
        stft = np.abs(librosa.stft(trimmed, n_fft=512, hop_length=256, win_length=512))
    else:
        stft = np.abs(librosa.stft(trimmed, n_fft=512, hop_length=320, win_length=480, center=False))

    return stft


def create_feature_data(audio_folder, folder_name="features_new", trim_flag=True, same_training=True):
    os.makedirs(folder_name, exist_ok=True)
    list_data_audio = glob.glob(f"{audio_folder}/**/*.wav", recursive=True)
    step = 100
    for fa in list_data_audio:
        feature = get_STFT(fa, trim_flag=trim_flag, same_training=same_training)
        feature = feature.T
        file_name = fa.split("\\")[-1]
        dir_name = os.path.dirname(fa)

        os.makedirs(os.path.join(folder_name, dir_name), exist_ok=True)

        for i in range(0, feature.shape[0], step):
            new_feature = feature[i:i + step]

            mfccs = librosa.feature.mfcc(y=new_feature, sr=16000, n_mfcc=13)
            mfccs = np.mean(mfccs.T, axis=0)
            mfccs = mfccs.flatten()

            pad = pad_features(mfccs, target_length)

            np.save(f"{folder_name}/{file_name}_{i}.npy", pad)


def convert_keras2qttflite(feature_folder, keras_model_file, tflite_model_file):
    def representative_dataset_test():
        list_np = glob.glob(f"{feature_folder}/**/*.npy", recursive=True)
        for np_f in list_np:
            np_data = np.load(np_f)
            np_data = np.array(np_data, dtype=np.float32, ndmin=2)
            yield [np.expand_dims(np_data, axis=0)]

    keras_model = load_model(keras_model_file)
    # Convert the model to the TensorFlow Lite format with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_test
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # tf.lite.constants.INT8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    open(tflite_model_file, "wb").write(tflite_model)


def compare_model_kerasandtflite(audio_file, keras_model_file, tflite_model_file, trim_flag=True, same_training=False):
    feature = get_STFT(audio_file, trim_flag=trim_flag, same_training=same_training)
    feature = feature.T
    model_keras = load_model(keras_model_file)
    step = 10
    for i in range(0, feature.shape[0], step):

        new_feature = feature[i:i + step]

        mfccs = librosa.feature.mfcc(y=new_feature, sr=16000, n_mfcc=13)
        mfccs = np.mean(mfccs.T, axis=0)
        mfccs = mfccs.flatten()

        pad = pad_features(mfccs, target_length)

        new_feature = pad

        pred = model_keras.predict(np.asarray([new_feature]), verbose=0)

        interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        test_audio = new_feature

        if input_details["dtype"] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            test_audio = test_audio / input_scale + input_zero_point
            test_audio = np.around(test_audio)

        test_audio = test_audio.astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], [test_audio])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]

        output_type = output_details["dtype"]
        if output_type == np.uint8:
            output_scale, output_zero_point = output_details["quantization"]
            output = output_scale * (output.astype(np.float32) - output_zero_point)
        out_label = ""
        if np.argmax(output) == 0:
            out_label = "Alarm"
        elif np.argmax(output) == 2:
            out_label = "Water"
        else:
            out_label = "Other"
        print(f"\noutput:{i} / uint8 tflite-> {output}: keras-> {pred[0]}, label: {out_label}")


if __name__ == "__main__":
    audio_folder = "calibrate_data"
    feature_folder = "features_new"
    
    if not os.path.exists("input"):
        os.makedirs("input")

    audio_file = os.path.join("input", "water-audio.wav")
    keras_model_file = os.path.join("weights", "model_16k_1v2.h5")
    tflite_model_file = os.path.join("weights", "model_16k_quantized_1v2.tflite")

    if not os.path.exists(tflite_model_file):
        pass
    else:
        print(f"\nthe same file name already exists: {tflite_model_file}")
        sys.exit("error !")

    same_training = True
    trim_flag = True
    # Create feature data numpy from audio
    """ same_training to make data feature same with training proceed.
    audio_folder: should be small engough each type class its number is 10 files"""
    print("\n\n\n====================== Create feature data ======================\n\n\n")
    create_feature_data(audio_folder, feature_folder, trim_flag=trim_flag, same_training=same_training)

    print("\n\n\n====================== Convert keras to qt tflite model ======================\n\n\n")
    """ # Do convert model from keras to quantized tflite """
    convert_keras2qttflite(feature_folder, keras_model_file, tflite_model_file)

    print("\n\n\n====================== Compare keras to qt tflite model ======================\n\n\n")
    """ # # Compare model with origin """
    compare_model_kerasandtflite(audio_file, keras_model_file, tflite_model_file, trim_flag=trim_flag, same_training=same_training)
