import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="tensorflow_extra")
def smelu(x, beta=1.5):
    """Smooth ReLU (SmeLU): Smooth activations and reproducibility in deep networks, https://arxiv.org/abs/2010.09931

    Args:
        x : numpy or tensorflow tensor
        beta (float): smooth value. Defaults to 1.5.

    Returns:
        tensorflow tensor
    """
    x = tf.convert_to_tensor(x)
    return tf.where(
        tf.math.abs(x) <= beta, ((x + beta) ** 2) / (4 * beta), tf.nn.relu(x)
    )
