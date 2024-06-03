import os
import sys
import datetime
import numpy as np
import h5py
import random
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import librosa
from tqdm.auto import tqdm

def augment_data(data):
    augmented_data = []
    for sample in data:
        if random.random() > 0.5:
            sample = sample + random.uniform(-0.1, 0.1)  # Random noise
        if random.random() > 0.5:
            sample = sample * random.uniform(0.8, 1.2)  # Random scaling
        augmented_data.append(sample)
    return np.array(augmented_data)

def extract_features(file_path):
    mfccs = librosa.feature.mfcc(y=np.load(file_path), sr=16000, n_mfcc=13)  # Compute MFCCs
    mfccs_mean = np.mean(mfccs.T, axis=0)  # Compute the mean of the MFCCs
    return mfccs_mean.flatten()

def get_data(path):
    alamSound = ['Alarm', 'Alarm_clock', 'Bell', 'Busy_signal', 'Car_alarm', 'Cellphone_buzz_vibrating_alert',
                 'Fire_alarm', 'Siren', 'smoke_detector_smoke_alarm', 'Telephone_bell_ringing', 'audioset-smoke_alarm']
    waterSound = ['bathtub_filling_or_washing', 'Fill_with_liquid', 'Shower', 'Stream_river', 'toilet_flush',
                  'Water', 'Water_tap_faucet', 'Waterfall']
    otherSound = ['dishes_pots_and_pans', 'microwave_oven', 'Inside large room or hall', 'Inside public space', 'Other sourceless', 'Bird']

    X_train, Y_train, X_test, Y_test, X_validation, Y_validation = [], [], [], [], [], []
    label_counter = {'alarm': 0, 'other': 0, 'waterSound': 0}

    files = os.listdir(path)
    for file in tqdm(files, total=len(files), desc='Arranging Data'):
        file_path = os.path.join(path, file)
        features = extract_features(file_path)
        label = '_'.join(file.split('_')[1:]).split(".")[0]
        if label in alamSound:
            label = "alarm"
        elif label in waterSound:
            label = "waterSound"
        elif label in otherSound:
            label = "other"
        else:
            continue

        if label_counter[label] % 10 < 7:
            X_train.append(features)
            Y_train.append(label)
        elif label_counter[label] % 10 < 9:
            X_validation.append(features)
            Y_validation.append(label)
        else:
            X_test.append(features)
            Y_test.append(label)

        label_counter[label] += 1

    return (np.array(X_train), np.array(Y_train),
            np.array(X_validation), np.array(Y_validation),
            np.array(X_test), np.array(Y_test))

def print_M(conf_M, lb):
    s = "activity," + ",".join(lb.inverse_transform(range(len(conf_M))))
    print(s)
    for i, row in enumerate(conf_M):
        print(lb.inverse_transform([i])[0] + "," + ",".join(map(str, row)))

def print_M_P(conf_M, lb):
    s = "activity," + ",".join(lb.inverse_transform(range(len(conf_M))))
    print(s)
    for i, row in enumerate(conf_M):
        total = sum(row)
        percentages = [str(round(val / total, 2)) for val in row]
        print(lb.inverse_transform([i])[0] + "," + ",".join(percentages))

def showResult(result, y_test, lb):
    predictions = [np.argmax(y) for y in result]
    expected = [np.argmax(y) for y in y_test]

    num_labels = y_test.shape[1]
    conf_M = [[0] * num_labels for _ in range(num_labels)]

    for e, p in zip(expected, predictions):
        conf_M[e][p] += 1

    print_M(conf_M, lb)
    print_M_P(conf_M, lb)

def build_improved_model(input_shape, num_labels):
    model = Sequential()
    model.add(Dense(512, input_shape=(input_shape,), kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model

def load_weight(path):
    model = load_model(path)
    print(model.summary())
    return model

if __name__ == "__main__":
    
    model_weight_out = os.path.join('weights', 'model_16k_1v2.h5')
    if os.path.exists(model_weight_out):
        sys.exit(f"The same file name exists already: {model_weight_out}")

    featuresPath = "features_16k/"
    X_train, Y_train, X_validation, Y_validation, X_test, Y_test = get_data(featuresPath)

    lb = LabelEncoder()
    y_train = to_categorical(lb.fit_transform(Y_train))
    y_test = to_categorical(lb.fit_transform(Y_test))
    y_validation = to_categorical(lb.fit_transform(Y_validation))
    num_labels = y_train.shape[1]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    X_test = scaler.transform(X_test)

    X_train = augment_data(X_train)

    model = build_improved_model(X_train.shape[1], num_labels)
    model.summary()

    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return float(lr * tf.math.exp(-0.1))

    callback = LearningRateScheduler(scheduler)

    model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_validation, y_validation), callbacks=[callback])
    # model = load_weight("weights/model_16k_1v2.h5")
    print("\nmodel input structure: \n", model.input)

    result = model.predict(X_test)

    cnt, cnt_alarm, cnt_other, cnt_water = 0, 0, 0, 0
    alarm_num, other_num, water_num = sum(np.argmax(y_test, axis=1) == 0), sum(np.argmax(y_test, axis=1) == 1), sum(np.argmax(y_test, axis=1) == 2)

    for i in range(len(Y_test)):
        pred = np.argmax(result[i])
        if np.argmax(y_test[i]) == pred:
            cnt += 1
            if pred == 0:
                cnt_alarm += 1
            elif pred == 1:
                cnt_other += 1
            else:
                cnt_water += 1

    acc = round(cnt * 100 / len(Y_test), 2)
    acc_alarm = round(cnt_alarm * 100 / alarm_num, 2)
    acc_other = round(cnt_other * 100 / other_num, 2)
    acc_water = round(cnt_water * 100 / water_num, 2)

    print(f"Total Accuracy: {acc}%, Alarm Accuracy: {acc_alarm}%, Others Accuracy: {acc_other}%, Water Accuracy: {acc_water}%")

    showResult(result, y_test, lb)

    if not os.path.exists("Models"):
        os.makedirs("Models")
    path = os.path.join("Models", f"audio_NN_New{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_acc_{acc}")
    model_json = model.to_json()
    with open(f"{path}.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(f"{path}.weights.h5")

    if not os.path.exists("weights"):
        os.makedirs("weights")
    model.save(model_weight_out, overwrite=True, include_optimizer=False)