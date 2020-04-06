import pandas as pd
import tensorflow as tf
import numpy as np
from generate_batches_by_swo import generate_batches_by_sample

df = pd.read_csv('kc1.csv')
dataset = tf.data.Dataset.from_tensor_slices(df.values)
dataset = generate_batches_by_sample(dataset, 50, 40).repeat()

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
  dataset,
  steps_per_epoch = 40,
  epochs = 5
)

