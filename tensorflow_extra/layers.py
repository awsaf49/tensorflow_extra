import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_extra.utils import random_int, random_float

@tf.keras.utils.register_keras_serializable(package='tensorflow_extra')
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
    def __init__(self, n_fft=2048, hop_length=512, win_length=None, window='hann_window', 
                 sr=16000, n_mels=128, fmin=20.0, fmax=None, power_to_db=True, top_db=80.0, 
                 power=2.0, out_channels=None, name='mel_spectrogram', **kwargs):
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
        self.out_channels = out_channels

    def call(self, input):
        spec = self.spectrogram(input) # audio to spectrogram with shape
        spec = self.melscale(spec) # spectrogram to mel spectrogram
        if self.power_to_db:
            spec = self.dbscale(spec) # mel spectrogram to decibel mel spectrogram
        spec = tf.linalg.matrix_transpose(spec) # (..., time, n_mels) to (..., n_mels, time)
        if self.out_channels is not None:
            spec = self.update_channels(spec)
        return spec

    def spectrogram(self, input):
        spec = tf.signal.stft(input,
                              frame_length=self.win_length or self.n_fft,
                              frame_step=self.hop_length,
                              fft_length=self.n_fft,
                              window_fn=getattr(tf.signal, self.window),
                              pad_end=True)
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
        log_spec = 10.0 * (tf.math.log(input) / tf.math.log(10.0))
        log_spec = tf.math.maximum(log_spec, tf.math.reduce_max(log_spec) - self.top_db)
        return log_spec
    
    def update_channels(self, input):
        spec = input[..., tf.newaxis]
        if self.out_channels>1:
            multiples = tf.concat([tf.ones(tf.rank(spec), dtype=tf.int32), 
                                tf.constant([self.out_channels], dtype=tf.int32)], axis=0)
            spec = tf.tile(spec, multiples)
        return spec

    def get_config(self):
        config = super(MelSpectrogram, self).get_config()
        config.update({
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'win_length': self.win_length,
            'window': self.window,
            'sr': self.sr,
            'n_mels': self.n_mels,
            'fmin': self.fmin,
            'fmax': self.fmax,
            'power_to_db': self.power_to_db,
            'top_db': self.top_db,
            'power': self.power,
            'out_channels': self.out_channels,
        })
        return config