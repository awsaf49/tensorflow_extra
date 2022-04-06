# Tensorflow Extra
> Extra Utilities for Tensorflow

# Installation
```shell
pip install tensorflow-extra
```

# Activations
## SmeLU: Smooth ReLU
```py
import tensorflow as tf
import tensorflow_extra as tfe

a = tf.constant([-2.5, -1.0, 0.5, 1.0, 2.5])
b = tfe.activations.smelu(a)  # array([0., 0.04166667, 0.6666667 , 1.0416666 , 2.5])
```
<img src="images/smelu.png" width=500>