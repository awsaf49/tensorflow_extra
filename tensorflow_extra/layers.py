import tensorflow as tf
import tensorflow_probability as tfp


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
        **kwargs,
    ):
        super(MelSpectrogram, self).__init__(name=name, **kwargs)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window = window
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or int(sr / 2)
        self.power_to_db = power_to_db
        self.top_db = top_db
        self.power = power
        self.amin = amin
        self.ref = ref
        self.out_channels = out_channels

    @tf.function
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
            frame_length=self.win_length,
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
                    tf.ones(tf.rank(spec) - 1, dtype=tf.int32),
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

    @tf.function
    def call(self, images, labels=None, training=False):

        # Skip batch if not training or if prob is not met or if labels are not provided
        if tf.random.uniform([]) > self.prob or not training or labels is None:
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
        **kwargs,
    ):
        super(CutMix, self).__init__(name=name, **kwargs)
        self.alpha = alpha
        self.prob = prob
        self.full_height = full_height
        self.full_width = full_width

    @tf.function
    def call(self, images, labels=None, training=False):
        # Skip batch if not training or if prob is not met or if labels are not provided
        if tf.random.uniform([]) > self.prob or not training or labels is None:
            return (images, labels) if labels is not None else images

        # Ensure 4D input
        images, was_2d = self._ensure_4d(images)

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
        r_x = (
            tf.random.uniform([], maxval=W, dtype=tf.int32)
            if not self.full_width
            else 0
        )
        r_y = (
            tf.random.uniform([], maxval=H, dtype=tf.int32)
            if not self.full_height
            else 0
        )
        r = 0.5 * tf.math.sqrt(1.0 - lam)
        r_w_p = r if not self.full_width else 1.0
        r_h_p = r if not self.full_height else 1.0
        r_w_half = tf.cast(r_w_p * tf.cast(W, tf.float32), tf.int32)
        r_h_half = tf.cast(r_h_p * tf.cast(H, tf.float32), tf.int32)

        # Find the coordinates of the patch
        x1 = tf.cast(tf.clip_by_value(r_x - r_w_half, 0, W), tf.int32)
        x2 = tf.cast(tf.clip_by_value(r_x + r_w_half, 1, W), tf.int32)
        y1 = tf.cast(tf.clip_by_value(r_y - r_h_half, 0, H), tf.int32)
        y2 = tf.cast(tf.clip_by_value(r_y + r_h_half, 1, H), tf.int32)

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

        # Ensure original shape
        images = self._ensure_original_shape(images, was_2d)

        return images, labels

    def _ensure_4d(self, tensor):
        if len(tensor.shape) == 2:
            tensor = tf.expand_dims(tensor, axis=1)
            tensor = tf.expand_dims(tensor, axis=-1)
            return tensor, True
        return tensor, False

    def _ensure_original_shape(self, tensor, was_2d):
        if was_2d:
            tensor = tf.squeeze(tensor, axis=-1)
            tensor = tf.squeeze(tensor, axis=1)
        return tensor

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


@tf.keras.utils.register_keras_serializable(package="tensorflow_extra")
class TimeFreqMask(tf.keras.layers.Layer):
    """
    Applies Time Freq Mask to spectrogram input
    Ref: https://pytorch.org/audio/main/_modules/torchaudio/functional/functional.html#mask_along_axis_iid
    """

    def __init__(
        self,
        freq_mask_prob=0.5,
        num_freq_masks=2,
        freq_mask_param=10,
        time_mask_prob=0.5,
        num_time_masks=2,
        time_mask_param=20,
        time_last=True,
        name="time_freq_mask",
        **kwargs,
    ):
        super(TimeFreqMask, self).__init__(name=name, **kwargs)
        self.freq_mask_prob = freq_mask_prob
        self.num_freq_masks = num_freq_masks
        self.freq_mask_param = freq_mask_param
        self.time_mask_prob = time_mask_prob
        self.num_time_masks = num_time_masks
        self.time_mask_param = time_mask_param
        self.time_last = time_last

    @tf.function
    def call(self, inputs, training=False):
        if not training:
            return inputs
        x = inputs
        # Adjust input shape
        ndims = tf.rank(x)
        shape = tf.shape(x)

        #         if ndims == 3:
        #             x = x[tf.newaxis, ...]
        #             x = tf.reshape(x, shape=(1, tf.split(shape, 3)))
        #         elif ndims == 2:
        #             x = x[tf.newaxis, ..., tf.newaxis]
        #             x = tf.reshape(x, shape=(1, tf.split(shape, 2), 1))
        #         else:
        #             pass
        #         elif ndims > 4 or ndims < 2:
        #             raise ValueError("Input tensor must be 2, 3, or 4-dimensional.")
        # Apply time mask
        for _ in tf.range(self.num_time_masks):
            x = self.mask_along_axis_iid(
                x,
                self.time_mask_param,
                0,
                2 + int(self.time_last),
                self.time_mask_prob,
            )
        # Apply freq mask
        for _ in tf.range(self.num_freq_masks):
            x = self.mask_along_axis_iid(
                x,
                self.freq_mask_param,
                0,
                2 + int(not self.time_last),
                self.freq_mask_prob,
            )
        # Re-adjust output shape
        #         if ndims == 3:
        #             x = x[0]
        #         elif ndims == 2:
        #             x = x[0, ..., 0]
        return x

    def mask_along_axis_iid(self, specs, mask_param, mask_value, axis, p):
        if axis not in [2, 3]:
            raise ValueError("Only Frequency and Time masking are supported")

        if not 0.0 <= p <= 1.0:
            raise ValueError(f"The value of p must be between 0.0 and 1.0 ({p} given).")

        mask_param = mask_param # self._get_mask_param(mask_param, p, specs.shape[axis])
        if tf.random.uniform([]) > p:
            return specs

        specs = tf.transpose(specs, perm=[0, 3, 1, 2])  # (batch, channel, freq, time)

        dtype = specs.dtype
        shape = tf.shape(specs)

        value = tf.random.uniform(shape=shape[:2], dtype=dtype) * mask_param
        min_value = tf.random.uniform(shape=shape[:2], dtype=dtype) * (
            specs.shape[axis] - value
        )

        # Create broadcastable mask
        mask_start = tf.cast(min_value, tf.float32)[..., None, None]
        mask_end = (tf.cast(min_value, tf.float32) + tf.cast(value, tf.float32))[
            ..., None, None
        ]
        mask = tf.range(0, specs.shape[axis], dtype=dtype)

        # Per batch example masking
        specs = tf.linalg.matrix_transpose(specs) if axis == 2 else specs
        cond = (mask >= mask_start) & (mask < mask_end)
        specs = tf.where(
            cond, tf.fill(tf.shape(specs), tf.cast(mask_value, dtype=dtype)), specs
        )
        specs = tf.linalg.matrix_transpose(specs) if axis == 2 else specs

        specs = tf.transpose(specs, perm=[0, 2, 3, 1])  # (batch, freq, time, channel)

        return specs

    def get_config(self):
        config = super(TimeFreqMask, self).get_config()
        config.update(
            {
                "freq_mask_prob": self.freq_mask_prob,
                "num_freq_masks": self.num_freq_masks,
                "freq_mask_param": self.freq_mask_param,
                "time_mask_prob": self.time_mask_prob,
                "num_time_masks": self.num_time_masks,
                "time_mask_param": self.time_mask_param,
                "time_last": self.time_last,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="tensorflow_extra")
class ZScoreMinMax(tf.keras.layers.Layer):
    def __init__(self, name="z_score_min_max", **kwargs):
        super().__init__(name=name, **kwargs)

    @tf.function
    def call(self, inputs):
        # Standardize using Z-score
        mean = tf.math.reduce_mean(inputs)
        std = tf.math.reduce_std(inputs)
        standardized = tf.where(tf.math.equal(std, 0), inputs - mean, (inputs - mean) / std)

        # Normalize using Min-Max
        min_val = tf.math.reduce_min(standardized)
        max_val = tf.math.reduce_max(standardized)
        normalized = tf.where(tf.math.equal(max_val - min_val, 0), standardized - min_val,
                              (standardized - min_val) / (max_val - min_val))

        return normalized
    
    def get_config(self):
        config = super().get_config()
        return config