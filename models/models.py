import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


def bias_err(y_true, y_pred):
    return tf.reduce_sum(y_true) - tf.reduce_sum(y_pred)

def abs_bias_err(y_true, y_pred):
    return tf.abs(tf.reduce_sum(y_true) - tf.reduce_sum(y_pred))

def peak_err(y_true, y_pred):
    return tf.reduce_max(y_true) - tf.reduce_max(y_pred)

def abs_peak_err(y_true, y_pred):
    return tf.abs(tf.reduce_max(y_true) - tf.reduce_max(y_pred))

# residual convolutional block
def res_conv_block(x, kernelsize=3, filters=32, dropout=0, batchnorm=False, initializer='he_normal'):
    conv1 = layers.Conv2D(filters, (kernelsize, kernelsize), kernel_initializer=initializer, padding='same')(x)
    if batchnorm is True:
        conv1 = layers.BatchNormalization(axis=3)(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv2 = layers.Conv2D(filters, (kernelsize, kernelsize), kernel_initializer=initializer, padding='same')(conv1)
    if batchnorm is True:
        conv2 = layers.BatchNormalization(axis=3)(conv2)
        conv2 = layers.Activation("relu")(conv2)
    if dropout > 0:
        conv2 = layers.Dropout(dropout)(conv2)

    #skip connection
    shortcut = layers.Conv2D(filters, kernel_size=(1, 1), kernel_initializer=initializer, padding='same')(x)
    if batchnorm is True:
        shortcut = layers.BatchNormalization(axis=3)(shortcut)
    shortcut = layers.Activation("relu")(shortcut)
    respath = layers.add([shortcut, conv2])
    return respath

# gating signal for attention unit
def gatingsignal(input, out_size, batchnorm=False, initializer='he_normal'):
    x = layers.Conv2D(out_size, (1, 1), kernel_initializer=initializer, padding='same')(input)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

# attention unit/block based on soft attention
def attention_block(x, gating, inter_shape, initializer='he_normal'):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), kernel_initializer=initializer, padding='same')(x)
    shape_theta_x = K.int_shape(theta_x)
    phi_g = layers.Conv2D(inter_shape, (1, 1), kernel_initializer=initializer, padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3), strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]), kernel_initializer='he_normal', padding='same')(phi_g)
    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), kernel_initializer=initializer, padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)
    upsample_psi = layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': shape_x[3]})(upsample_psi)
    y = layers.multiply([upsample_psi, x])
    result = layers.Conv2D(shape_x[3], (1, 1), kernel_initializer=initializer, padding='same')(y)
    attenblock = layers.BatchNormalization()(result)
    return attenblock

def create_masks(input_shape=(96, 96), r1=7, r2=35):
    # Define the center of the circle
    center_x, center_y = input_shape[1] // 2, input_shape[0] // 2

    # Create a grid of x and y coordinates
    Y, X = tf.meshgrid(tf.range(input_shape[0]), tf.range(input_shape[1]), indexing='ij')

    # Calculate the distance of each point from the center
    dist_from_center = tf.sqrt(tf.square(tf.cast(X, tf.float32) - center_x) + tf.square(tf.cast(Y, tf.float32) - center_y))

    # Create masks based on the distances
    mask_eye = dist_from_center < r1
    mask_eyewall = (dist_from_center < r2) & (dist_from_center > r1)
    mask_outer = dist_from_center > r2

    # Cast masks to float
    mask_eye = tf.cast(mask_eye, tf.float32)
    mask_eyewall = tf.cast(mask_eyewall, tf.float32)
    mask_outer = tf.cast(mask_outer, tf.float32)

    # Ensure masks are in the correct format for broadcasting
    mask_eye = tf.reshape(mask_eye, [1, input_shape[0], input_shape[1], 1])
    mask_eyewall = tf.reshape(mask_eyewall, [1, input_shape[0], input_shape[1], 1])
    mask_outer = tf.reshape(mask_outer, [1, input_shape[0], input_shape[1], 1])

    return mask_eye, mask_eyewall, mask_outer

# Example of using these masks in a network (u

def create_region_specific_branches(input_tensor, filters):
    mask_eye, mask_eyewall, mask_outer = create_masks()
    center_region = input_tensor * mask_eye
    ring_region = input_tensor * mask_eyewall 
    outer_region = input_tensor * mask_outer

    # Process each region separately
    center_output = res_conv_block(center_region, filters=filters)
    ring_output = res_conv_block(ring_region, filters=filters * 2)  # Assume more filters for more complex region
    outer_output = res_conv_block(outer_region, filters=filters)

    # Combine outputs
    combined = layers.concatenate([center_output, ring_output, outer_output])
    final_output = res_conv_block(combined, filters=filters)
    return final_output

# arunet
def arunet_threeblock(input_shape,
                           filters,
                           loss_func,
                           dropout=0.0,
                           batchnorm=True,
                           initializer='he_normal'):

    kernelsize = 3
    upsample_size = 2

    inputs = layers.Input(input_shape)

    # Downsampling layers
    dn_1 = res_conv_block(inputs, kernelsize, filters[0], dropout, batchnorm, initializer)
    pool1 = layers.MaxPooling2D(pool_size=(2,2))(dn_1)

    dn_2 = res_conv_block(pool1, kernelsize, filters[1], dropout, batchnorm, initializer)
    pool2 = layers.MaxPooling2D(pool_size=(2,2))(dn_2)

    dn_3 = res_conv_block(pool2, kernelsize, filters[2], dropout, batchnorm, initializer)
    pool3 = layers.MaxPooling2D(pool_size=(2,2))(dn_3)

    dn_4 = res_conv_block(pool3, kernelsize, filters[3], dropout, batchnorm, initializer)

    # Upsampling layers
    gating_4 = gatingsignal(dn_4, filters[2], batchnorm, initializer=initializer)
    att_4 = attention_block(dn_3, gating_4, filters[2], initializer=initializer)
    up_4 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(dn_4)
    up_4 = layers.concatenate([up_4, att_4], axis=3)
    up_conv_4 = res_conv_block(up_4, kernelsize, filters[2], dropout, batchnorm, initializer)

    gating_3 = gatingsignal(up_conv_4, filters[1], batchnorm, initializer=initializer)
    att_3 = attention_block(dn_2, gating_3, filters[1], initializer=initializer)
    up_3 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_4)
    up_3 = layers.concatenate([up_3, att_3], axis=3)
    up_conv_3 = res_conv_block(up_3, kernelsize, filters[1], dropout, batchnorm, initializer)

    gating_2 = gatingsignal(up_conv_3, filters[0], batchnorm, initializer=initializer)
    att_2 = attention_block(dn_1, gating_2, filters[0], initializer=initializer)
    up_2 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_3)
    up_2 = layers.concatenate([up_2, att_2], axis=3)
    up_conv_2 = res_conv_block(up_2, kernelsize, filters[0], dropout, batchnorm, initializer)

    conv_final = layers.Conv2D(1, kernel_size=(1,1), kernel_initializer=initializer)(up_conv_2)
    outputs = layers.Activation('relu')(conv_final)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=loss_func,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

# arunet
def arunet_threeblock_REGIONS(input_shape,
                           filters,
                           loss_func,
                           dropout=0.0,
                           batchnorm=True,
                           initializer='he_normal'):

    kernelsize = 3
    upsample_size = 2

    inputs = layers.Input(input_shape)

    # Downsampling layers
    dn_1 = res_conv_block(inputs, kernelsize, filters[0], dropout, batchnorm, initializer)
    pool1 = layers.MaxPooling2D(pool_size=(2,2))(dn_1)

    dn_2 = res_conv_block(pool1, kernelsize, filters[1], dropout, batchnorm, initializer)
    pool2 = layers.MaxPooling2D(pool_size=(2,2))(dn_2)

    dn_3 = res_conv_block(pool2, kernelsize, filters[2], dropout, batchnorm, initializer)
    pool3 = layers.MaxPooling2D(pool_size=(2,2))(dn_3)

    dn_4 = res_conv_block(pool3, kernelsize, filters[3], dropout, batchnorm, initializer)

    # Upsampling layers
    gating_4 = gatingsignal(dn_4, filters[2], batchnorm, initializer=initializer)
    att_4 = attention_block(dn_3, gating_4, filters[2], initializer=initializer)
    up_4 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(dn_4)
    up_4 = layers.concatenate([up_4, att_4], axis=3)
    up_conv_4 = res_conv_block(up_4, kernelsize, filters[2], dropout, batchnorm, initializer)

    gating_3 = gatingsignal(up_conv_4, filters[1], batchnorm, initializer=initializer)
    att_3 = attention_block(dn_2, gating_3, filters[1], initializer=initializer)
    up_3 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_4)
    up_3 = layers.concatenate([up_3, att_3], axis=3)
    up_conv_3 = res_conv_block(up_3, kernelsize, filters[1], dropout, batchnorm, initializer)

    gating_2 = gatingsignal(up_conv_3, filters[0], batchnorm, initializer=initializer)
    att_2 = attention_block(dn_1, gating_2, filters[0], initializer=initializer)
    up_2 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_3)
    up_2 = layers.concatenate([up_2, att_2], axis=3)
    up_conv_2 = res_conv_block(up_2, kernelsize, filters[0], dropout, batchnorm, initializer)

    region_specific_output = create_region_specific_branches(up_conv_2, filters[0])

    conv_final = layers.Conv2D(1, kernel_size=(1,1), kernel_initializer=initializer)(region_specific_output)
    outputs = layers.Activation('relu')(conv_final)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=loss_func,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


# arunet
def arunet_fourblock(input_shape,
                           filters,
                           loss_func,
                           dropout=0.0,
                           batchnorm=True,
                           initializer='he_normal'):

    kernelsize = 3
    upsample_size = 2

    inputs = layers.Input(input_shape)

    # Downsampling layers
    dn_1 = res_conv_block(inputs, kernelsize, filters[0], dropout, batchnorm, initializer)
    pool1 = layers.MaxPooling2D(pool_size=(2,2))(dn_1)

    dn_2 = res_conv_block(pool1, kernelsize, filters[1], dropout, batchnorm, initializer)
    pool2 = layers.MaxPooling2D(pool_size=(2,2))(dn_2)

    dn_3 = res_conv_block(pool2, kernelsize, filters[2], dropout, batchnorm, initializer)
    pool3 = layers.MaxPooling2D(pool_size=(2,2))(dn_3)

    dn_4 = res_conv_block(pool3, kernelsize, filters[3], dropout, batchnorm, initializer)
    pool4 = layers.MaxPooling2D(pool_size=(2,2))(dn_4)

    dn_5 = res_conv_block(pool4, kernelsize, filters[4], dropout, batchnorm, initializer)

    # Upsampling layers
    gating_5 = gatingsignal(dn_5, filters[3], batchnorm, initializer=initializer)
    att_5 = attention_block(dn_4, gating_5, filters[3], initializer=initializer)
    up_5 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(dn_5)
    up_5 = layers.concatenate([up_5, att_5], axis=3)
    up_conv_5 = res_conv_block(up_5, kernelsize, filters[3], dropout, batchnorm, initializer)

    gating_4 = gatingsignal(up_conv_5, filters[2], batchnorm, initializer=initializer)
    att_4 = attention_block(dn_3, gating_4, filters[2], initializer=initializer)
    up_4 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_5)
    up_4 = layers.concatenate([up_4, att_4], axis=3)
    up_conv_4 = res_conv_block(up_4, kernelsize, filters[2], dropout, batchnorm, initializer)

    gating_3 = gatingsignal(up_conv_4, filters[1], batchnorm, initializer=initializer)
    att_3 = attention_block(dn_2, gating_3, filters[1], initializer=initializer)
    up_3 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_4)
    up_3 = layers.concatenate([up_3, att_3], axis=3)
    up_conv_3 = res_conv_block(up_3, kernelsize, filters[1], dropout, batchnorm, initializer)

    gating_2 = gatingsignal(up_conv_3, filters[0], batchnorm, initializer=initializer)
    att_2 = attention_block(dn_1, gating_2, filters[0], initializer=initializer)
    up_2 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_3)
    up_2 = layers.concatenate([up_2, att_2], axis=3)
    up_conv_2 = res_conv_block(up_2, kernelsize, filters[0], dropout, batchnorm, initializer)

    conv_final = layers.Conv2D(1, kernel_size=(1,1), kernel_initializer=initializer)(up_conv_2)
    outputs = layers.Activation('relu')(conv_final)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=loss_func,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


# #Residual-Attention UNET (RA-UNET)
# def arunet_fourblock(input_shape,
#                            filters,
#                            loss_func,
#                            dropout=0.0,
#                            batchnorm=True):

#     kernelsize = 3
#     upsample_size = 2

#     inputs = layers.Input(input_shape)

#     # Downsampling layers
#     dn_1 = res_conv_block(inputs, kernelsize, filters[0], dropout, batchnorm)
#     pool1 = layers.MaxPooling2D(pool_size=(2,2))(dn_1)

#     dn_2 = res_conv_block(pool1, kernelsize, filters[1], dropout, batchnorm)
#     pool2 = layers.MaxPooling2D(pool_size=(2,2))(dn_2)

#     dn_3 = res_conv_block(pool2, kernelsize, filters[2], dropout, batchnorm)
#     pool3 = layers.MaxPooling2D(pool_size=(2,2))(dn_3)

#     dn_4 = res_conv_block(pool3, kernelsize, filters[3], dropout, batchnorm)
#     pool4 = layers.MaxPooling2D(pool_size=(2,2))(dn_4)

#     dn_5 = res_conv_block(pool4, kernelsize, filters[4], dropout, batchnorm)

#     # Upsampling layers
#     gating_5 = gatingsignal(dn_5, filters[3], batchnorm)
#     att_5 = attention_block(dn_4, gating_5, filters[3])
#     up_5 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(dn_5)
#     up_5 = layers.concatenate([up_5, att_5], axis=3)
#     up_conv_5 = res_conv_block(up_5, kernelsize, filters[3], dropout, batchnorm)

#     gating_4 = gatingsignal(up_conv_5, filters[2], batchnorm)
#     att_4 = attention_block(dn_3, gating_4, filters[2])
#     up_4 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_5)
#     up_4 = layers.concatenate([up_4, att_4], axis=3)
#     up_conv_4 = res_conv_block(up_4, kernelsize, filters[2], dropout, batchnorm)

#     gating_3 = gatingsignal(up_conv_4, filters[1], batchnorm)
#     att_3 = attention_block(dn_2, gating_3, filters[1])
#     up_3 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_4)
#     up_3 = layers.concatenate([up_3, att_3], axis=3)
#     up_conv_3 = res_conv_block(up_3, kernelsize, filters[1], dropout, batchnorm)

#     gating_2 = gatingsignal(up_conv_3, filters[0], batchnorm)
#     att_2 = attention_block(dn_1, gating_2, filters[0])
#     up_2 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_3)
#     up_2 = layers.concatenate([up_2, att_2], axis=3)
#     up_conv_2 = res_conv_block(up_2, kernelsize, filters[0], dropout, batchnorm)

#     conv_final = layers.Conv2D(1, kernel_size=(1,1))(up_conv_2)
#     conv_final = layers.BatchNormalization(axis=3)(conv_final)
#     outputs = layers.Activation('sigmoid')(conv_final)

#     model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
#     model.compile(optimizer=RMSprop(learning_rate=0.0001),
#                   loss=loss_func,
#                   metrics=['mean_absolute_error'])
#     return model
