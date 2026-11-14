from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model, optimizers
import urllib.request

weights_url = "https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_file = "inception_v3.h5"
urllib.request.urlretrieve(weights_url, weights_file)
pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3), include_top=False, weights=None
)
pre_trained_model.load_weights(weights_file)

# pre_trained_model.summary()

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer("mixed7")
last_output = last_layer.output
print("last layer output shape: ", last_output.shape)

# add our dense layer
# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation="relu")(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation="sigmoid")(x)

# Define the model with pre-trained model following by our dense layer
model = Model(pre_trained_model.input, x)
model.compile(
    optimizer=optimizers.RMSprop(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["acc"],
)

# Connect with our data generator

training_dir = "horse-or-human/training/"
validation_dir = "horse-or-human/validation/"

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1 / 255)
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(150, 150),  # image pixels size
    class_mode="binary",  # binary: two categories, categorical: more than two categories
)

validation_datagen = ImageDataGenerator(rescale=1 / 255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(150, 150), class_mode="binary"
)

history = model.fit(train_generator, epochs=15, validation_data=validation_generator)
