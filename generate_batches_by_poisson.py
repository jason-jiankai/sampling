import pandas as pd
import tensorflow as tf
import numpy as np

def generate_batches_by_sample(dataset, batch_size, batch_num):
    dataset_size = len(list(dataset.as_numpy_iterator()))
    ratio = float(batch_size) / dataset_size
    
    batch = []
    for _ in range(batch_num):
        batch_features, batch_labels = [], []
        for e in dataset.as_numpy_iterator():
            if np.random.uniform(0,1) < ratio:
                batch_features.append(e[:-1])
                batch_labels.append(e[-1])
        batch.append((tf.convert_to_tensor(batch_features), tf.convert_to_tensor(batch_labels)))
    
    features_shape = [None, batch[0][0].shape[1]]
    labels_shape = [None, ]
    dataset = tf.data.Dataset.from_generator(
        lambda: batch,
        (tf.float64, tf.float64),
        (tf.TensorShape(features_shape), tf.TensorShape(labels_shape))
    )
    return dataset

