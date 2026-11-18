import tensorflow as tf
from profiling.tracing import label
import tensorflow_datasets as tfds

# Common form
data = tfds.load("cats_vs_dogs", split="train", as_supervised=True)


# the first 10k records
data = tfds.load("cats_vs_dogs", split="train[:10000]", as_supervised=True)

# the %20 of the records
data = tfds.load("cats_vs_dogs", split="train[:20%]", as_supervised=True)

# first 10k and last 10k records
data = tfds.load("cats_vs_dogs", split="train[-1000:]+train[:1000]", as_supervised=True)

# Custom splits:
train_data = tfds.load("cats_vs_dogs", split="train[:80%]", as_supervised=True)
validation_data = tfds.load("cats_vs_dogs", split="train[80%:90%]", as_supervised=True)
test_data = tfds.load("cats_vs_dogs", split="train[-10%:]", as_supervised=True)

# Only for debugging
train_length = [i for i, _ in enumerate(train_data)][-1] + 1
print(train_length)
