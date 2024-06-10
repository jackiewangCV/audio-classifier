import tensorflow as tf
import numpy as np
from create_dataset import create_dataset

DATASET_DIR     = '/home/rudolf/dev/nomo/dataset/alarmset-custom'
CLASS_TO_ID_MAP = {'other': 0, 'alarm': 1}
BATCH_SIZE      = 32
train_dataset, validation_dataset, class_weights = create_dataset(DATASET_DIR, CLASS_TO_ID_MAP, BATCH_SIZE)

train_dataset = train_dataset.unbatch();
def representative_dataset():
    for data in train_dataset.batch(1).take(100):
        yield [tf.expand_dims(data[0], axis=-1)]

# Export the model in "saved model" format since TFLite conversion directly from the ".keras" format is flaky.
# Direct conversion should be considered in the future.
model                               = tf.keras.models.load_model('model.keras')
model.export('saved_model')
converter                           = tf.lite.TFLiteConverter.from_saved_model('saved_model')
# converter                           = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations             = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset    = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type      = tf.uint8
converter.inference_output_type     = tf.uint8
tflite_quant_model                  = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_quant_model)

# ------------------------------------- KERAS MODEL
def split_wav(wav):
    N_SAMPLES = 16000
    n = int(tf.size(wav) / N_SAMPLES)
    audio = wav[:(n * N_SAMPLES)]
    return tf.reshape(tensor=audio, shape=(n, N_SAMPLES)) # Reshape along batch dim

keras_model  = tf.keras.models.load_model('model')

from create_dataset import get_log_mel_spectrogram, load_wav_16k_mono, preprocess

file         = '/home/rudolf/dl/jason-nomo_dataset/audio_dataset/Water/y (99).wav'
wav          = load_wav_16k_mono(file)
wav          = split_wav(wav)
spectrograms = tf.stack([get_log_mel_spectrogram(preprocess(wav_second)) for wav_second in wav])
y_pred       = keras_model.predict(spectrograms)
y_pred
THRESH = 0.95
np.argmax(y_pred > THRESH, axis=1)

# --------------- TFLITE MODEL
interpreter  = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
my_signature = interpreter.get_signature_runner()
my_signature._signature_def
validation_dataset.unbatch().as_numpy_iterator().next()[0].shape
input = validation_dataset.unbatch().as_numpy_iterator().next()[0]
quantized = tf.quantization.quantize(input, -128.0, 127.0, tf.quint8)[0]
quantized = quantized[tf.newaxis, ...]
output = my_signature(input_2=quantized)
output
