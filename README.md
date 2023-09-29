# Tensorflow Extra
> TensorFlow GPU & TPU compatible operations: MelSpectrogram, TimeFreqMask, CutMix, MixUp, ZScore, and more

# Installation
For Stable version
```shell
!pip install tensorflow-extra
```
or
For updated version
```shell
!pip install git+https://github.com/awsaf49/tensorflow_extra
```
# Layers
## MelSpectrogram
Converts audio data to mel-spectrogram in GPU/TPU.
```py
import tensorflow_extra as tfe
audio2spec = tfe.layers.MelSpectrogram()
spec = audio2spec(audio)
```

<img src="https://github.com/awsaf49/tensorflow_extra/assets/36858976/45981a3f-fe32-423b-9a0d-5016b8463bbf" width="600">


## Time Frequency Masking
Can also control number of stripes.
```py
time_freq_mask = tfe.layers.TimeFreqMask()
spec = time_freq_mask(spec)
```
<img src="https://github.com/awsaf49/tensorflow_extra/assets/36858976/78bc7007-67e1-4a93-8f26-9d8a2e687edd" width="600">

## CutMix
Can be used with audio, spec, image. For spec full freq resolution can be used using `full_height=True`.
```py
cutmix = tfe.layers.CutMix()
audio = cutmix(audio, training=True) # accepts both audio & spectrogram
```
<img src="https://github.com/awsaf49/tensorflow_extra/assets/36858976/35af3140-46ec-4592-8923-4bd21f76cb15" width="600">


## MixUp
Can be used with audio, spec, image. For spec full freq resolution can be used using `full_height=True`.
```py
mixup = tfe.layers.MixUp()
audio = mixup(audio, training=True)  # accepts both audio & spectrogram
```

<img src="https://github.com/awsaf49/tensorflow_extra/assets/36858976/128de4aa-5295-4655-b00d-1e16b5e06560" width="600">


## Normalization
Applies standardization and rescaling.
```py
norm = tfe.layers.ZScoreMinMax()
spec = norm(spec)
```
<img src="https://github.com/awsaf49/tensorflow_extra/assets/36858976/8a8a4b38-9eb2-4dda-ab09-11887b37c593" width="600">


# Activations
## SmeLU: Smooth ReLU
```py
import tensorflow as tf
import tensorflow_extra as tfe

a = tf.constant([-2.5, -1.0, 0.5, 1.0, 2.5])
b = tfe.activations.smelu(a)  # array([0., 0.04166667, 0.6666667 , 1.0416666 , 2.5])
```
<img src="images/smelu.png" width=500>
