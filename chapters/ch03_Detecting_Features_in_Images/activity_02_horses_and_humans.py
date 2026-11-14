from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


training_dir = "horse-or-human/training/"
validation_dir = "horse-or-human/validation/"

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1 / 255)
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),  # image pixels size
    class_mode="binary",  # binary: two categories, categorical: more than two categories
)

validation_datagen = ImageDataGenerator(rescale=1 / 255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(300, 300), class_mode="binary"
)

# Too many layers to get smaller images
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            16, (3, 3), activation="relu", input_shape=(300, 300, 3)
        ),  # 3 x 3 x 3 filter plus bias, 16 filters => 448 parameters to learn
        tf.keras.layers.MaxPooling2D(2, 2),
        # input (3 * 3 * 16) + 1 = 145 (3x3 filters, 16 filters, plus bias)
        # then 32 * 145 = 4640 parameters to learn
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
        tf.keras.layers.Dense(
            1, activation="sigmoid"
        ),  # we have two categories, so we use sigmoid activation function
    ]
)

model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    metrics=["accuracy"],
)
# lr: learning rate, the step size at each iteration while moving toward a minimum of a loss function

history = model.fit(train_generator, epochs=15, validation_data=validation_generator)

# until now we get 99% accuaracy but 88% validation accuaracy
# which means we have overfitting problem.
# We need to use data augmentation to solve this problem.
