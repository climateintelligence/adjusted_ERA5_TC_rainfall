import numpy as np # type: ignore
import tensorflow as tf # type: ignore
import keras.backend as K # type: ignore
import math as m
import tensorflow_probability as tfp # type: ignore
import tensorflow_addons as tfa # type: ignore

"""
FSS functions
"""

def filter_normalise(data,  filter_shape=(3, 3), sigma=1):
    normalised = (data-K.min(data))/(K.max(data)-K.min(data) + tf.keras.backend.epsilon())
    tf.debugging.check_numerics(normalised, "normalised 1 contains NaN or infinity")
    filtered = tfa.image.gaussian_filter2d(data, filter_shape=filter_shape, sigma=sigma)
    normalised = (filtered-K.min(filtered))/(K.max(filtered)-K.min(filtered))
    return normalised

def activation_arctan(x):
    epsilon = 1e-7
    activation = epsilon + (1 - 2 * epsilon) * (0.5 +  tf.math.atan(x)/tf.constant(m.pi))
    noise = tf.random.uniform(tf.shape(activation), minval=-epsilon, maxval=epsilon)
    activation += noise
    return activation

def produce_binary_fields(rainfall_field,threshold): #the threshold could be either physical or a percentile
    # where the thresholds is exceed you get 1, otherwise 0
    binary_rainfall = activation_arctan(rainfall_field - threshold)
    return binary_rainfall

def generating_fractions_filtered(y_true, y_pred, threshold_true, threshold_pred,n): #generating fractions applied to both forecast and observations on a given neighbourhood of size n and a given threshold
    # n must be an odd number (it is the number of grid points along the neighbourhood side
    binary_true = produce_binary_fields(y_true,threshold_true)
    binary_true = filter_normalise(binary_true)
    binary_pred = produce_binary_fields(y_pred,threshold_pred)
    binary_pred = filter_normalise(binary_pred)

    kernel_in = tf.constant(np.ones((n, n, 1, 1)), dtype=tf.float32) / (n*n)

    true_fractions = tf.nn.conv2d(input=binary_true, filters=kernel_in, strides=[1, 1, 1, 1], padding='SAME') #x_in is your input
    pred_fractions = tf.nn.conv2d(input=binary_pred, filters=kernel_in, strides=[1, 1, 1, 1], padding='SAME')
    return true_fractions,pred_fractions

def fractions_skill_score_filtered(y_true, y_pred, threshold_true, threshold_pred, n):  #this is to compute the metrics applied to the fractions
    true_fractions,pred_fractions=generating_fractions_filtered(y_pred,y_true,threshold_true,threshold_pred,n)
    num = tf.experimental.numpy.nanmean(K.square(pred_fractions - true_fractions))
    denom = tf.experimental.numpy.nanmean(K.square(pred_fractions) + K.square(true_fractions))
    return num,denom

def fss_loss_filtered(y_true, y_pred, n, q):
    # NOTE: n defines the number of grid-points for the neighbourhood side, it's an odd number (1,3,5,7,...)
    # y_pred should be regridded to y_true first
    threshold_true = tfp.stats.percentile(x=y_true, q=q) + tf.keras.backend.epsilon()
    threshold_pred = tfp.stats.percentile(x=y_pred, q=q) + tf.keras.backend.epsilon()
    num,denom=fractions_skill_score_filtered(y_true, y_pred, threshold_true, threshold_pred, n)
    fss=num/(denom + tf.keras.backend.epsilon())
    return fss


"""
Compound loss
"""

def compound_loss_fss(loss_weights):
    def _compound_loss(y_true, y_pred):
        # apply fss_loss_filtered to each item in the batch
        def _apply_fss_loss_single(y_true_single, y_pred_single):
            # reshape the inputs
            y_true_single = tf.expand_dims(y_true_single, axis=0)
            y_pred_single = tf.expand_dims(y_pred_single, axis=0)

            fss_80 = fss_loss_filtered(y_true=y_true_single, y_pred=y_pred_single, n=15, q=80)
            fss_95 = fss_loss_filtered(y_true=y_true_single, y_pred=y_pred_single, n=15, q=95)
            fss_99 = fss_loss_filtered(y_true=y_true_single, y_pred=y_pred_single, n=15, q=99)
            return fss_80 + fss_95 + fss_99

        # fss loss component
        fss_loss = tf.map_fn(lambda y: (_apply_fss_loss_single(y[0], y[1]),), (y_true, y_pred), dtype=(tf.float32,))

        # mae loss component
        mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)(y_true, y_pred)
        total_loss = (loss_weights[0] * tf.reduce_mean(fss_loss) + 
                      loss_weights[1] * mse_loss)
        return total_loss
    return _compound_loss
