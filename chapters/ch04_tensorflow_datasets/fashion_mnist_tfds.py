import tensorflow as tf
import tensorflow_datasets as tfds


# batch_size=-1 means load the entire dataset into memory,
# which is useful for small datasets like Fashion MNIST.
# The as_supervised=True argument returns the data in a
# (image, label) format, which is convenient for training models.
(training_images, training_labels), (test_images, test_labels) = tfds.as_numpy(
    tfds.load(
        "fashion_mnist", split=["train", "test"], batch_size=-1, as_supervised=True
    )
)

training_images = training_images / 255.0
test_images = test_images / 255.0


model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(training_images, training_labels, epochs=5)
