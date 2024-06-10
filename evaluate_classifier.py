import tensorflow as tf

def compute_recall(y_true, y_pred):
   recall = tf.keras.metrics.Recall()
   recall.update_state(y_true, y_pred)
   return recall.result().numpy()

def compute_precision(y_true, y_pred):
   precision = tf.keras.metrics.Precision()
   precision.update_state(y_true, y_pred)
   return precision.result().numpy()

def compute_f1(y_true, y_pred):
   precision, recall = compute_precision(y_true, y_pred), compute_recall(y_true, y_pred)
   return 2 * (precision * recall) / (precision + recall)

tf.math.argmax(tf.convert_to_tensor([0.5, 0.3, 0.2])).numpy()

compute_precision([0, 0, 1], [0, 1, 1])
compute_recall([1, 0, 1], [0, 1, 1])
compute_f1([1, 0, 1], [0, 1, 1])

N_LABELS = 3
y_true = tf.convert_to_tensor([1, 2, 0, 2], tf.int32)
y_true_one_hot = tf.one_hot(y_true, N_LABELS)
y_pred = tf.convert_to_tensor([[0.2, 0.8, 0.0],
                               [0.1, 0.5, 0.4],
                               [0.1, 0.6, 0.1],
                               [0.8, 0.0, 0.2]], tf.float32)

def compute_f1_per_class(y_true, y_pred, n_labels):
    y_true_one_hot = tf.one_hot(y_true, n_labels)
    f1 = tf.keras.metrics.F1Score()
    f1.update_state(y_true_one_hot, y_pred)
    return f1.result()

compute_f1_per_class(y_true, y_pred, N_LABELS)
