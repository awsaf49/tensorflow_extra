import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_extra.utils import random_int, random_float


@tf.keras.utils.register_keras_serializable(package="tensorflow_extra")
class MelSpectrogram(tf.keras.layers.Layer):
    """
    Mel Spectrogram Layer to convert audio to mel spectrogram which works with single or batched inputs.

    Args:
        n_fft (int): Size of the FFT window.
        hop_length (int): Number of samples between successive STFT columns.
        win_length (int): Size of the STFT window. If None, defaults to n_fft.
        window_fn (str): Name of the window function to use.
        sr (int): Sample rate of the input signal.
        n_mels (int): Number of mel bins to generate.
        fmin (float): Minimum frequency of the mel bins.
        fmax (float): Maximum frequency of the mel bins. If None, defaults to sr / 2.
        power (float): Exponent for the magnitude spectrogram.
        power_to_db (bool): Whether to convert the power spectrogram to decibels.
        top_db (float): Maximum decibel value for the output spectrogram.
        power_to_db (bool): Whether to convert spectrogram from energy to power.
        out_channels (int): Number of output channels. If None, no channel is created.

    Call Args:
        input (tf.Tensor): Audio signal of shape (audio_len,) or (None, audio_len)

    Returns:
        tf.Tensor: Mel spectrogram of shape (..., n_mels, time, out_channels)
        or (..., n_mels, time) if out_channels is None.

    """

    def __init__(
        self,
        n_fft=2048,
        hop_length=512,
        win_length=None,
        window="hann_window",
        sr=16000,
        n_mels=128,
        fmin=20.0,
        fmax=None,
        power_to_db=True,
        top_db=80.0,
        power=2.0,
        amin=1e-10,
        ref=1.0,
        out_channels=None,
        name="mel_spectrogram",
        **kwargs
    ):
        super(MelSpectrogram, self).__init__(name=name, **kwargs)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sr / 2
        self.power_to_db = power_to_db
        self.top_db = top_db
        self.power = power
        self.amin = amin
        self.ref = ref
        self.out_channels = out_channels

    def call(self, input):
        spec = self.spectrogram(input)  # audio to spectrogram with shape
        spec = self.melscale(spec)  # spectrogram to mel spectrogram
        if self.power_to_db:
            spec = self.dbscale(spec)  # mel spectrogram to decibel mel spectrogram
        spec = tf.linalg.matrix_transpose(
            spec
        )  # (..., time, n_mels) to (..., n_mels, time)
        if self.out_channels is not None:
            spec = self.update_channels(spec)
        return spec

    def spectrogram(self, input):
        spec = tf.signal.stft(
            input,
            frame_length=self.win_length or self.n_fft,
            frame_step=self.hop_length,
            fft_length=self.n_fft,
            window_fn=getattr(tf.signal, self.window),
            pad_end=True,
        )
        spec = tf.math.pow(tf.math.abs(spec), self.power)
        return spec

    def melscale(self, input):
        nbin = tf.shape(input)[-1]
        matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=nbin,
            sample_rate=self.sr,
            lower_edge_hertz=self.fmin,
            upper_edge_hertz=self.fmax,
        )
        return tf.tensordot(input, matrix, axes=1)

    def dbscale(self, input):
        log_spec = 10.0 * (
            tf.math.log(tf.math.maximum(input, self.amin)) / tf.math.log(10.0)
        )
        if callable(self.ref):
            ref_value = self.ref(log_spec)
        else:
            ref_value = tf.math.abs(self.ref)
        log_spec -= (
            10.0
            * tf.math.log(tf.math.maximum(ref_value, self.amin))
            / tf.math.log(10.0)
        )
        log_spec = tf.math.maximum(log_spec, tf.math.reduce_max(log_spec) - self.top_db)
        return log_spec

    def update_channels(self, input):
        spec = input[..., tf.newaxis]
        if self.out_channels > 1:
            multiples = tf.concat(
                [
                    tf.ones(tf.rank(spec), dtype=tf.int32),
                    tf.constant([self.out_channels], dtype=tf.int32),
                ],
                axis=0,
            )
            spec = tf.tile(spec, multiples)
        return spec

    def get_config(self):
        config = super(MelSpectrogram, self).get_config()
        config.update(
            {
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "win_length": self.win_length,
                "window": self.window,
                "sr": self.sr,
                "n_mels": self.n_mels,
                "fmin": self.fmin,
                "fmax": self.fmax,
                "power_to_db": self.power_to_db,
                "top_db": self.top_db,
                "power": self.power,
                "amin": self.amin,
                "ref": self.ref,
                "out_channels": self.out_channels,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="tensorflow_extra")
class MixUp(tf.keras.layers.Layer):
    """
    MixUp Augmentation Layer to apply MixUp to one batch.

    Args:
        alpha (float): Alpha parameter for beta distribution.
        prob (float): Probability of applying MixUp.

    Call Args:
        images (tf.Tensor): Batch of images.
        labels (tf.Tensor): Batch of labels.

    Returns:
        tf.Tensor: Batch of image.
        tf.Tensor: Batch of labels.

    """

    def __init__(self, alpha=0.2, prob=0.5, name="mix_up", **kwargs):
        super(MixUp, self).__init__(name=name, **kwargs)
        self.alpha = alpha
        self.prob = prob

    def call(self, images, labels=None, training=False):

        # Skip batch if not training or if prob is not met or if labels are not provided
        if random_float() > self.prob or not training or labels is None:
            return (images, labels) if labels is not None else images

        # Get original shape
        spec_shape = tf.shape(images)
        label_shape = tf.shape(labels)

        # Select lambda from beta distribution
        beta = tfp.distributions.Beta(self.alpha, self.alpha)
        lam = beta.sample(1)[0]

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        images = lam * images + (1 - lam) * tf.roll(images, shift=1, axis=0)
        labels = lam * labels + (1 - lam) * tf.roll(labels, shift=1, axis=0)

        # Ensure original shape
        images = tf.reshape(images, spec_shape)
        labels = tf.reshape(labels, label_shape)

        return images, labels

    def get_config(self):
        config = super(MixUp, self).get_config()
        config.update(
            {
                "alpha": self.alpha,
                "prob": self.prob,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="tensorflow_extra")
class CutMix(tf.keras.layers.Layer):
    """
    Augmentation layer to apply CutMix to one batch.

    Args:
        alpha (float): Alpha parameter for beta distribution.
        prob (float): Probability of applying CutMix.
        full_height (bool): If True, the patch will be cut with full height of the image.
        full_width (bool): If True, the patch will be cut with full width of the image.

    Call Args:
        images (tf.Tensor): Batch of images.
        labels (tf.Tensor): Batch of labels.

    Returns:
        tf.Tensor: Batch of image.
        tf.Tensor: Batch of labels.
    """

    def __init__(
        self,
        alpha=0.2,
        prob=0.5,
        full_height=False,
        full_width=False,
        name="cut_mix",
        **kwargs
    ):
        super(CutMix, self).__init__(name=name, **kwargs)
        self.alpha = alpha
        self.prob = prob
        self.full_height = full_height
        self.full_width = full_width

    def call(self, images, labels=None, training=False):

        # Skip batch if not training or if prob is not met or if labels are not provided
        if random_float() > self.prob or not training or labels is None:
            return (images, labels) if labels is not None else images

        # Get original shapes
        image_shape = tf.shape(images)
        label_shape = tf.shape(labels)

        # Select lambda from beta distribution
        beta = tfp.distributions.Beta(self.alpha, self.alpha)
        lam = beta.sample(1)[0]

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        images_rolled = tf.roll(images, shift=1, axis=0)
        labels_rolled = tf.roll(labels, shift=1, axis=0)

        # Find dimensions of patch
        H = tf.cast(image_shape[1], tf.int32)
        W = tf.cast(image_shape[2], tf.int32)
        r_x = random_int([], minval=0, maxval=W) if not self.full_width else 0
        r_y = random_int([], minval=0, maxval=H) if not self.full_height else 0
        r = 0.5 * tf.math.sqrt(1.0 - lam)
        r_w_p = r if not self.full_width else 1.0
        r_h_p = r if not self.full_height else 1.0
        r_w_half = tf.cast(r_w_p * tf.cast(W, tf.float32), tf.int32)
        r_h_half = tf.cast(r_h_p * tf.cast(H, tf.float32), tf.int32)

        # Find the coordinates of the patch
        x1 = tf.cast(tf.clip_by_value(r_x - r_w_half, 0, W), tf.int32)
        x2 = tf.cast(tf.clip_by_value(r_x + r_w_half, 0, W), tf.int32)
        y1 = tf.cast(tf.clip_by_value(r_y - r_h_half, 0, H), tf.int32)
        y2 = tf.cast(tf.clip_by_value(r_y + r_h_half, 0, H), tf.int32)

        # Extract outer-pad patch -> [0, 0, 1, 1, 0, 0]
        patch1 = images[:, y1:y2, x1:x2, :]  # [batch, height, width, channel]
        patch1 = tf.pad(
            patch1, [[0, 0], [y1, H - y2], [x1, W - x2], [0, 0]]
        )  # outer-pad

        # Extract inner-pad patch -> [2, 2, 0, 0, 2, 2]
        patch2 = images_rolled[:, y1:y2, x1:x2, :]
        patch2 = tf.pad(
            patch2, [[0, 0], [y1, H - y2], [x1, W - x2], [0, 0]]
        )  # outer-pad
        patch2 = images_rolled - patch2  # inner-pad = img - outer-pad

        # Combine patches [0, 0, 1, 1, 0, 0] + [2, 2, 0, 0, 2, 2] -> [2, 2, 1, 1, 2, 2]
        images = patch1 + patch2

        # Combine labels
        lam = tf.cast((1.0 - (x2 - x1) * (y2 - y1) / (W * H)), tf.float32)
        labels = lam * labels + (1.0 - lam) * labels_rolled

        # Ensure original shape
        images = tf.reshape(images, image_shape)
        labels = tf.reshape(labels, label_shape)

        return images, labels

    def get_config(self):
        config = super(CutMix, self).get_config()
        config.update(
            {
                "alpha": self.alpha,
                "prob": self.prob,
                "full_height": self.full_height,
                "full_width": self.full_width,
            }
        )
        return config
