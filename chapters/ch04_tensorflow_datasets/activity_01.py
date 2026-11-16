import tensorflow as tf
import tensorflow_datasets as tfds

mnist_data, info = tfds.load(name="fashion_mnist", with_info="true")
for item in mnist_data:
    print(item)

# Get the training and test datasets
# mnist_train = tfds.load(name="fashion_mnist", split="train")
# or
mnist_train = mnist_data["train"]

assert isinstance(mnist_train, tf.data.Dataset)
print(type(mnist_train))  # DatasetAdapter

# Inspect the first element of the dataset
for item in mnist_train.take(1):
    print(type(item))
    print(item.keys())
    # print(item["image"])
    # print(item["label"])
    # print(item["image"].shape)
    # print(item["label"].numpy())  # get the label as a numpy array
print(info)  # DatasetInfo object
