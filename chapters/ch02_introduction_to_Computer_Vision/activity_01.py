import keras


import tensorflow as tf

data = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

# This is a normalization step, it converts the pixel values from 0-255 to 0-1.
training_images = training_images / 255.0
test_images = test_images / 255.0


# Flatten layer converts the 2D image into a 1D array of pixels.
# Activation function relu (rectified linear unit) returns the input if it is positive, otherwise it returns 0.
# Activation function softmax determines which neuron has the max probability.
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

# Explore model output
classifications = model.predict(test_images)
print(classifications[500])
print(test_labels[500])
