import tensorflow as tf
from profiling.tracing import label
import tensorflow_datasets as tfds

# another option is to use tensorflow-addons
# but versions are not always compatible with the latest tensorflow version
# pip install tensorflow==2.15.0 tensorflow-addons==0.23.0


def augmentimages(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255
    image = tf.image.random_flip_left_right(image)
    return image, label


data = tfds.load("horses_or_humans", split="train", as_supervised=True)
train = data.map(augmentimages)
train_batches = train.shuffle(100).batch(10)


val_data = tfds.load("horses_or_humans", split="test", as_supervised=True)
validation_batches = val_data.batch(32)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            16, (3, 3), activation="relu", input_shape=(300, 300, 3)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(
    train_batches, epochs=10, validation_data=validation_batches, validation_steps=1
)
