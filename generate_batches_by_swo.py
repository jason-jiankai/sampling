import pandas as pd
import tensorflow as tf
import numpy as np

def generate_batches_by_sample(dataset, batch_size, steps):
    dataset_size = len(list(dataset.as_numpy_iterator()))
    batch_features, batch_labels = [], []
    data_features, data_labels = [], []

    for _ in range(steps):
        sampling = set(np.random.choice(dataset_size, size=batch_size, replace=False))
        i = 0
        for e in dataset.as_numpy_iterator():
            if i in sampling:
                batch_features.append(e[:-1])
                batch_labels.append(e[-1])
            i += 1
        data_features.append(batch_features)
        data_labels.append(batch_labels)
        batch_features, batch_labels = [], []

    dataset = tf.data.Dataset.from_tensor_slices((data_features, data_labels))
    return dataset

