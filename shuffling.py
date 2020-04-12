import pandas as pd
import tensorflow as tf

df = pd.read_csv('kc1.csv')
target = df.pop('defects')
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
print(len(df))

train_dataset = dataset.shuffle(len(df)).batch(50).repeat()
# print(list(train_dataset.take(1).as_numpy_iterator()))

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
  train_dataset,
  steps_per_epoch = 40,
  epochs = 5
)

