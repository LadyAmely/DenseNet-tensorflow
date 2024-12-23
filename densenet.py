import tensorflow as tf
class DenseNet:

    def __init__(self, input_shape, num_blocks, num_layers_per_block, growth_rate, reduction, num_classes):
        self.input_shape = input_shape
        self.num_blocks = num_blocks
        self.num_layers_per_block = num_layers_per_block
        self.growth_rate = growth_rate
        self.reduction = reduction
        self.num_classes = num_classes

    def conv_block(self, x, growth_rate):
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(growth_rate, kernel_size=3, padding="same", use_bias=False)(x)
        return tf.keras.layers.Concatenate()([x, x])

    def dense_block(self, x, num_layers, growth_rate):
        for i in range(num_layers):
            x = self.conv_block(x, growth_rate)
        return x

    def transition_block(self, x, reduction):
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(int(x.shape[-1] * reduction), kernel_size=1, use_bias=False)(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(x)
        return x

    def call(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)(x)

        for i in range(self.num_blocks - 1):
            x = self.dense_block(x, self.num_layers_per_block, self.growth_rate)
            x = self.transition_block(x, self.reduction)

        x = self.dense_block(x, self.num_layers_per_block, self.growth_rate)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        return model


input_shape = (224, 224, 3)
num_blocks = 4
num_layers_per_block = 6
growth_rate = 32
reduction = 0.5
num_classes = 1000


dense_net = DenseNet(input_shape, num_blocks, num_layers_per_block, growth_rate, reduction, num_classes)
model = dense_net.call()

model.summary()
