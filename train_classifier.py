import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from create_dataset import create_dataset

DATASET_DIR     = '/home/rudolf/dev/nomo/dataset/alarmset-custom'
CLASS_TO_ID_MAP = {'other': 0, 'alarm': 1}
BATCH_SIZE      = 32
train_dataset, validation_dataset, class_weights = create_dataset(DATASET_DIR, CLASS_TO_ID_MAP, BATCH_SIZE)

train_dataset.as_numpy_iterator().next()[0].shape

model = tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(60, 93, 1)),
      tf.keras.layers.Conv2D(32, 6, activation='relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(3,3)),
      tf.keras.layers.Conv2D(16, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(3,3)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(1, activation='sigmoid')
])
print(model.summary())

METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'), 
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.AUC(name='prc', curve='PR'),
]

model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss      = tf.keras.losses.BinaryCrossentropy(),
    metrics   = METRICS
)

EARLY_STOPPING = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc', 
    verbose=1,
    patience=10,
    start_from_epoch=15, 
    mode='max',
    restore_best_weights=True
)

history = model.fit(
    train_dataset,
    validation_data = validation_dataset,
    epochs          = 150,
    callbacks       = EARLY_STOPPING,
    class_weight    = class_weights
)

metrics = history.history
plt.figure(figsize=(16,6))
plt.subplot(3,3,1)
plt.plot(history.epoch, metrics['prc'], metrics['val_prc'])
plt.legend(['prc', 'val_prc'])
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('PRC')

plt.subplot(3,3,2)
plt.plot(history.epoch, 100 * np.array(metrics['accuracy']),
                        100 * np.array(metrics['val_accuracy']))
plt.legend(['accuracy', 'val_accuracy'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(3,3,3)
plt.plot(history.epoch, 100 * np.array(metrics['recall']),
                        100 * np.array(metrics['val_recall']))
plt.legend(['recall', 'val_recall'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Recall')

plt.subplot(3,3,4)
plt.plot(history.epoch, 100 * np.array(metrics['precision']),
                        100 * np.array(metrics['val_precision']))
plt.legend(['precision', 'val_precision'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Precision')

plt.subplot(3,3,5)
plt.plot(history.epoch, np.array(metrics['tp']),
                        np.array(metrics['val_tp']))
plt.legend(['tp', 'val_tp'])
plt.xlabel('Epoch')
plt.ylabel('TP')

plt.subplot(3,3,6)
plt.plot(history.epoch, np.array(metrics['fp']),
                        np.array(metrics['val_fp']))
plt.legend(['fp', 'val_fp'])
plt.xlabel('Epoch')
plt.ylabel('FP')

plt.subplot(3,3,7)
plt.plot(history.epoch, np.array(metrics['tn']),
                        np.array(metrics['val_tn']))
plt.legend(['tn', 'val_tn'])
plt.xlabel('Epoch')
plt.ylabel('TN')

plt.subplot(3,3,8)
plt.plot(history.epoch, np.array(metrics['fn']),
                        np.array(metrics['val_fn']))
plt.legend(['fn', 'val_fn'])
plt.xlabel('Epoch')
plt.ylabel('FN')

plt.show()

model.save('model.keras')
