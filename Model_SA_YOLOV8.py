import numpy as np
from tensorflow import keras
from keras import layers
import tensorflow as tf
from Classificaltion_Evaluation import net_evaluation

# Basic Convolution Block
def convolution_block(x, filters, kernel_size=3, strides=1, activation="silu", shortcut=True):
    y = layers.Conv2D(filters, kernel_size, strides=strides, padding="same")(x)
    y = layers.BatchNormalization()(y)
    if activation == "silu":
        y = layers.Activation(tf.nn.silu)(y)
    else:
        y = layers.Activation(activation)(y)
    if shortcut:
        if x.shape[-1] != y.shape[-1] or strides != 1:
            x = layers.Conv2D(filters, kernel_size=1, strides=strides, padding="same")(x)
            x = layers.BatchNormalization()(x)
        y = layers.Add()([x, y])
    return y

# C2F Block
def c2f_block(x, filters, n=3, shortcut=True):
    inputs = x
    split_channels = filters // 2
    layers_list = [convolution_block(inputs, split_channels, 1, 1, shortcut=False)]
    for _ in range(n):
        layers_list.append(
            convolution_block(layers_list[-1], split_channels, 3, 1, shortcut=False))
    x = layers.Concatenate()(layers_list)
    x = convolution_block(x, filters, 1, 1, shortcut=shortcut)
    return x

# SPPF Block
def sppf_block(x, filters, pool_size=5):
    x1 = layers.MaxPooling2D(pool_size=pool_size, strides=1, padding="same")(x)
    x2 = layers.MaxPooling2D(pool_size=pool_size, strides=1, padding="same")(x1)
    x3 = layers.MaxPooling2D(pool_size=pool_size, strides=1, padding="same")(x2)

    x = layers.Concatenate()([x, x1, x2, x3])
    x = convolution_block(x, filters, 1, 1)
    return x

# Self-Attention Block
def self_attention_block(x, num_heads=4, key_dim=32):
    # Flatten spatial dimensions
    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    x_reshaped = layers.Reshape((h * w, c))(x)
    attention = layers.MultiHeadAttention(num_heads=num_heads,key_dim=key_dim)(x_reshaped, x_reshaped)
    # Residual connection
    x_reshaped = layers.Add()([x_reshaped, attention])
    x_reshaped = layers.LayerNormalization()(x_reshaped)
    # Restore spatial dimensions
    x = layers.Reshape((h, w, c))(x_reshaped)
    return x

# SA-YOLOv8 Architecture
def SA_YOLOv8(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Backbone
    x = convolution_block(inputs, 32, 3, 1, shortcut=False)
    x = convolution_block(x, 64, 3, 2)
    x = c2f_block(x, 64)

    x = convolution_block(x, 128, 3, 2)
    x = c2f_block(x, 128)

    x = convolution_block(x, 256, 3, 2)
    x = c2f_block(x, 256)

    x = convolution_block(x, 512, 3, 2)
    x = c2f_block(x, 512)

    x = sppf_block(x, 512)

    # Self-Attention Layer Added Here
    x = self_attention_block(x)
    # Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model

# Main Wrapper Function
def Model_SA_YOLOV8(Data, Target):
    IMG_SIZE = 32
    NUM_CLASSES = 3

    Data = np.zeros((Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Data.shape[0]):
        temp = np.resize(Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Data[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))
    Target = np.zeros((Target.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Data.shape[0]):
        temp = np.resize(Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Target[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    model = SA_YOLOv8(IMG_SIZE, NUM_CLASSES)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",metrics=["accuracy"])

    model.fit(Data,Target, epochs=20, batch_size=16, verbose=2)

    pred = model.predict(Target)
    pred_labels = np.zeros_like(pred)
    pred_labels[np.arange(pred.shape[0]), np.argmax(pred, axis=1)] = 1
    Eval = [net_evaluation(Target[n].astype('uint8'), pred_labels[n].astype('uint8')) for n in range(pred.shape[0])]
    return Eval, pred


