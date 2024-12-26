import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

def preprocess_image(image, label):

    image = tf.expand_dims(image, axis=-1)
    image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, (64, 64))
    image = tf.cast(image, tf.float32) / 255.0

    return image, label

def load_and_preprocess_data(batch_size=64):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=1000)
        .cache()
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_ds, test_ds

train_ds, test_ds = load_and_preprocess_data(batch_size=64)
