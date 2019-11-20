"""
Contains various bits and pieces needed for the method described in the paper: 
Momentum Contrast for Unsupervised Visual Representation Learning 
(https://arxiv.org/abs/1911.05722)
"""

import tensorflow as tf


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


if __name__ == "__main__":
    x = tf.random.normal([1, 1, 1, 3])
    x = tf.tile(x, [8, 1, 1, 1])
    x = tf.concat([x[-4:], x[-4:] + tf.random.normal([4, 1, 1, 3]) * 0.05], 0)
    bn = EmulateMultiGPUBatchNorm(2, axis=3, momentum=0.0)
