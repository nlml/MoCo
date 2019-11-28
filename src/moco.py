"""
Contains various bits and pieces needed for the method described in the paper: 
Momentum Contrast for Unsupervised Visual Representation Learning 
(https://arxiv.org/abs/1911.05722)
"""

import tensorflow as tf


class MoCoQueue:
    def __init__(self, embedding_dim, max_queue_length):
        self.embedding_dim = embedding_dim
        # Put a single zeros key in there to start with, it will be pushed out eventually
        with tf.device("CPU:0"):
            self.keys = tf.random.normal([2, self.embedding_dim])
        self.max_queue_length = max_queue_length

    def enqueue(self, new_keys):
        self.keys = tf.concat([new_keys, self.keys], 0)
        if self.keys.shape[0] > self.max_queue_length:
            self.keys = self.keys[:self.max_queue_length]


@tf.function
def _moco_training_step_inner(x, x_aug, queue, model, model_ema, temperature, optimizer):
    N = tf.shape(x)[0]
    K = tf.shape(queue)[0]
    C = tf.shape(queue)[1]
    k = model_ema(x_aug, training=True)  # no gradient
    with tf.GradientTape() as tape:
        q = model(x, training=True)
        l_pos = tf.matmul(tf.reshape(q, [N, 1, C]), tf.reshape(k, [N, C, 1]))
        l_pos = tf.reshape(l_pos, [N, 1])
        l_neg = tf.matmul(tf.reshape(q, [N, C]), tf.reshape(queue, [C, K]))
        logits = tf.concat([l_pos, l_neg], axis=1)
        labels = tf.zeros([N], dtype="int64")
        loss = tf.reduce_mean(
            tf.losses.sparse_categorical_crossentropy(labels, logits / temperature)
        )
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, k


def moco_training_step(
    x,
    x_aug,
    queue,
    model,
    model_ema,
    optimizer,
    temperature=0.07,
    momentum=0.999,
):
    loss, new_keys = _moco_training_step_inner(x, x_aug, queue.keys, model, model_ema,
        tf.constant(temperature, dtype='float32'), optimizer)
    # update the EMA of the model
    update_model_via_ema(model, model_ema, momentum)
    queue.enqueue(new_keys)
    return loss


def update_model_via_ema(
    model, ema_model, momentum, just_trainable_vars=False
):
    iterable = (
        zip(model.trainable_variables, ema_model.trainable_variables)
        if just_trainable_vars
        else zip(model.variables, ema_model.variables)
    )
    for p, p2 in iterable:
        p2.assign(momentum * p2 + (1.0 - momentum) * p)


class EmulateMultiGPUBatchNorm(tf.keras.layers.Layer):
    """Emulates behaviour of batch norm when training on multi GPUs when only a single 
    GPU is being used. This technique is used in the paper (see heading 'Shuffling BN' 
    in Section 3.3)."""

    def __init__(self, num_gpus, axis=1, *args, **kwargs):
        if axis != 1 and axis != 3:
            raise NotImplementedError(
                "Currently, EmulateMultiGPUBatchNorm just supports axis==1 or axis==3"
            )
        super(EmulateMultiGPUBatchNorm, self).__init__()
        # Only can do axis==3 as otherwise will get error:
        # "InternalError: The CPU implementation of FusedBatchNorm only supports
        #  NHWC tensor format for now. [Op:FusedBatchNormV3]"
        self.bn_layer = tf.keras.layers.BatchNormalization(
            axis=3, *args, **kwargs
        )
        self.num_gpus = num_gpus
        self.axis = axis

    def call(self, inputs, training=None):
        # Either NHWC (means axis=3) or NCHW (means axis=1)
        # First, for reshaping, we need NCHW:
        if self.axis == 3:
            inputs = tf.transpose(inputs, [0, 3, 1, 2])
        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)
        reshaped_inputs = self._reshape_into_groups(
            inputs, input_shape, tensor_input_shape
        )
        normalized_inputs = self.bn_layer(reshaped_inputs, training=training)
        outputs = tf.reshape(normalized_inputs, tensor_input_shape)
        if self.axis == 3:
            outputs = tf.transpose(outputs, [0, 2, 3, 1])
        return outputs

    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):
        # N,C,H,W --> N // G, C * G, H, W
        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        group_shape[1] = group_shape[1] * self.num_gpus
        group_shape[0] = group_shape[0] // self.num_gpus
        group_shape = tf.stack(group_shape)
        reshaped_inputs = tf.reshape(inputs, group_shape)
        # Back to NHWC
        reshaped_inputs = tf.transpose(reshaped_inputs, [0, 2, 3, 1])
        return reshaped_inputs

    def get_moving_mean_and_var_for_regular_bn(self):
        return [
            tf.reduce_mean(tf.reshape(v, (2, -1)), 0)
            for v in (self.bn_layer.moving_mean, self.bn_layer.moving_variance)
        ]


# if __name__ == "__main__":
#     x = tf.random.normal([1, 1, 1, 3])
#     x = tf.tile(x, [8, 1, 1, 1])
#     x = tf.concat([x[-4:], x[-4:] + tf.random.normal([4, 1, 1, 3]) * 0.05], 0)
#     bn = EmulateMultiGPUBatchNorm(2, axis=3, momentum=0.0)
from moco import moco_training_step

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

from moco import MoCoQueue, update_model_via_ema

EMBEDDING_DIM = 64
queue = MoCoQueue(EMBEDDING_DIM, 256)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 127.5 - 1, x_test / 127.5 - 1
x_train, x_test = [tf.cast(i, 'float32') for i in (x_train, x_test)]

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(2)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(256, activation='relu')
        self.d2 = Dense(EMBEDDING_DIM, activation=None)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

# Create an instance of the model
model = MyModel()
model_ema = MyModel()

# Initialise the models and make the EMA model 90% similar to the main model
model(x_train[:1])
model_ema(x_train[:1])
update_model_via_ema(model, model_ema, 0.1)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

EPOCHS = 5

for epoch in range(EPOCHS):
    for x, y in train_ds:
        x_aug = x + 0.1 * tf.random.normal(x.shape, dtype='float32')
        moco_training_step(x, x_aug, queue, model, model_ema, optimizer)