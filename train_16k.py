import os
import sys
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, \
    Add, Input, Conv1D, MaxPooling1D, Flatten, \
    Lambda

from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.metrics import classification_report


def print_M(conf_M, class_names):
    s = "activity," + ",".join(class_names)
    print(s)
    for i, row in enumerate(conf_M):
        print(class_names[i] + "," + ",".join(map(str, row)))


def print_M_P(conf_M, class_names):
    s = "activity," + ",".join(class_names)
    print(s)
    for i, row in enumerate(conf_M):
        total = sum(row)
        percentages = [str(round(val / total, 2)) if total > 0 else '0' for val in row]
        print(class_names[i] + "," + ",".join(percentages))


def showResult(result, y_test, class_names):
    predictions = [np.argmax(y) for y in result]
    expected = [np.argmax(y) for y in y_test]

    num_labels = y_test.shape[1]
    conf_M = [[0] * num_labels for _ in range(num_labels)]

    for e, p in zip(expected, predictions):
        conf_M[e][p] += 1

    print_M(conf_M, class_names)
    print_M_P(conf_M, class_names)


def load_weight(path):
    model = load_model(path)
    print(model.summary())
    return model


def build_improved_model(input_shape, num_labels):
    model = tf.keras.models.Sequential([
        Input(shape=(input_shape, 1)),
        Conv1D(32, 6, activation='relu'),
        MaxPooling1D(pool_size=(3)),
        Conv1D(16, 3, activation='relu'),
        MaxPooling1D(pool_size=(3)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(18, activation='relu'),
        Dropout(0.5),
        Dense(num_labels, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def load_weight(path):
    model = load_model(path)
    print(model.summary())
    return model

if __name__ == "__main__":

    model_weight_out = os.path.join('weights', 'exp_model_16k_1.3.weights.h5')

    # if os.path.exists(model_weight_out):
    #     sys.exit(f"The same file name exists already: {model_weight_out}")

    ############### Loading the datasets #####################

    print('\nLoading the data\n')

    featuresPath = "features_16k/"

    class_names = np.load(os.path.join(featuresPath, 'class_names.npy'))

    X_train = np.load(os.path.join(featuresPath, 'X_train.npy'))
    y_train = np.load(os.path.join(featuresPath, 'y_train.npy'))

    X_val = np.load(os.path.join(featuresPath, 'X_val.npy'))
    y_val = np.load(os.path.join(featuresPath, 'y_val.npy'))

    X_test = np.load(os.path.join(featuresPath, 'X_test.npy'))
    y_test = np.load(os.path.join(featuresPath, 'y_test.npy'))

    num_labels = y_train.shape[1]

    print("\nBalancing the data\n")

    print("Train Class distribution before balancing:", Counter(np.argmax(y_train, axis=1)))

    # Upsampling using SMOTE
    smote = SMOTE(sampling_strategy={1: 12000, 2: 10000})
    oversampled_features, oversampled_labels = smote.fit_resample(X_train, y_train)

    # Downsampling using RandomUnderSampler
    undersampler = RandomUnderSampler(sampling_strategy={0: 7300})
    undersampled_features, undersampled_labels = undersampler.fit_resample(
        oversampled_features, oversampled_labels)

    print("Train Class distribution after balancing:", Counter(
        np.argmax(undersampled_labels, axis=1)))

    X_train = undersampled_features
    y_train = undersampled_labels

    ###################### Training the model ###########################3
    print("\nTraining the model\n")

    model = build_improved_model(X_train.shape[1], num_labels)

    # model.summary()

    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return float(lr * tf.math.exp(-0.1))


    callback = LearningRateScheduler(scheduler)

    model.fit(X_train, y_train, batch_size=5, epochs=10,
              validation_data=(X_val, y_val), callbacks=[callback])

    ################# Testing the model #############################
    print("\nTesting the model\n")

    result = model.predict(X_test)

    cnt, cnt_alarm, cnt_other, cnt_water = 0, 0, 0, 0
    alarm_num, other_num, water_num = (sum(np.argmax(y_test, axis=1) == 0),
                                       sum(np.argmax(y_test, axis=1) == 1),
                                       sum(np.argmax(y_test, axis=1) == 2))

    for i in range(len(y_test)):
        pred = np.argmax(result[i])
        if np.argmax(y_test[i]) == pred:
            cnt += 1
            if pred == 0:
                cnt_alarm += 1
            elif pred == 1:
                cnt_other += 1
            else:
                cnt_water += 1

    acc = round(cnt * 100 / len(y_test), 2)
    acc_alarm = round(cnt_alarm * 100 / alarm_num, 2)
    acc_other = round(cnt_other * 100 / other_num, 2)
    acc_water = round(cnt_water * 100 / water_num, 2)

    print(
        f"Total Accuracy: {acc}%, Alarm Accuracy: {acc_alarm}%, Others Accuracy: {acc_other}%, Water Accuracy: {acc_water}%")

    showResult(result, y_test, class_names)

    print("\n")
    print(classification_report(
        np.argmax(y_test, axis=1),
        np.argmax(result, axis=1),
        target_names=list(class_names)
    ))

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
