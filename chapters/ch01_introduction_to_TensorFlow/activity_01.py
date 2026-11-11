import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# Sequential defines layers, we have only one layer in this case
# Dense: the most common layer type, a fully connected layer.
# This layer has one unit (neuron) and input shape of 1

# Dense multiplies the input by a weight and adds a bias.
# The weight and bias are initialized randomly, and they will
# be adjusted during training to minimize the loss function.

# For input_shape=[1] we have y = mx + b,
# if input_shape=[2] we would have y = m1x1 + m2x2 + b, and so on.
l0 = Dense(units=1, input_shape=[1])
model = Sequential([l0])


# An optimizer is an algorithm that given the previous guess, loss and the gradient, it will give us a new guess.
# This minimize the loss function, which is a measure of how far off our predictions are from the actual values.
model.compile(optimizer="sgd", loss="mean_squared_error")

# Prepare data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train the model, the fit method establishes xs as the input and ys as the expected output.
# The model will adjust its weights and biases to minimize the loss function.
model.fit(xs, ys, epochs=500)

print(model.predict(np.array([10.0])))
print("Here is what I learned: {}".format(l0.get_weights()))
