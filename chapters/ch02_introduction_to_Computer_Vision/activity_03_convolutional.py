import tensorflow as tf

data = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

# This is a normalization step, it converts the pixel values from 0-255 to 0-1.
# reshape is used to change the shape of the data to fit the model input requirements.
# from 28 x 28 to 28 x 28 x 1, where 1 indicates that the image is grayscale (single channel).
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0


# Flatten layer converts the 2D image into a 1D array of pixels.
# Activation function relu (rectified linear unit) returns the input if it is positive, otherwise it returns 0.
# Activation function softmax determines which neuron has the max probability.

# 64 indicates the number of filters
# (3, 3) indicates the size of the filter
# input_shape=(28, 28, 1) indicates the shape of the input image (28x28 pixels with 1 color channel)

# (2,2) indicates the size of the pooling window
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),  # this only reduces the size
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            128, activation=tf.nn.relu
        ),  # this learn 3 x 3 filter: 9 parammeters plus bias = 10
        # so the total number of parameters in this layer is 10 x 64 = 640 for each filter
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.fit(training_images, training_labels, epochs=50)

model.evaluate(test_images, test_labels)

# Explore model output
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])

print(model.summary())
