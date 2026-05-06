import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np
from Classificaltion_Evaluation import net_evaluation

# Ghost Convolution Block
def GhostConv(x, filters, kernel_size=3, ratio=2, strides=1):

    init_filters = filters // ratio
    new_filters = filters - init_filters

    primary = Conv2D(init_filters, kernel_size, strides=strides,
                     padding='same', activation='relu')(x)

    cheap = DepthwiseConv2D(3, padding='same',
                            activation='relu')(primary)

    cheap = Conv2D(new_filters, 1, padding='same',
                   activation='relu')(cheap)

    out = Concatenate()([primary, cheap])
    return out

# Multiscale Gated Attention (MGA)
def MultiScaleGatedAttention(x):

    # Multi-scale feature extraction
    k1 = Conv2D(x.shape[-1], 1, padding="same", activation="relu")(x)
    k3 = Conv2D(x.shape[-1], 3, padding="same", activation="relu")(x)
    k5 = Conv2D(x.shape[-1], 5, padding="same", activation="relu")(x)

    multi = Concatenate()([k1, k3, k5])

    # Gating mechanism
    gap = GlobalAveragePooling2D()(multi)
    gate = Dense(x.shape[-1], activation="sigmoid")(gap)
    gate = Reshape((1, 1, x.shape[-1]))(gate)
    out = Multiply()([x, gate])

    return out

# GELAN Block (Generalized Efficient Layer Aggregation)
def GELAN_Block(x, filters, n=3):
    route = GhostConv(x, filters, 1)
    outputs = [route]
    y = route
    for _ in range(n):
        y = GhostConv(y, filters, 3)
        outputs.append(y)
    concat = Concatenate()(outputs)
    fused = Conv2D(filters, 1, padding='same',
                   activation='relu')(concat)
    return fused

# Backbone: GCY9 with GELAN + MGA
def GCY9_GELAN_Backbone(inputs):

    # Stage 1
    x1 = GhostConv(inputs, 32, 3, strides=1)
    x1 = MultiScaleGatedAttention(x1)
    x1 = GELAN_Block(x1, 32, n=2)

    # Stage 2
    x2 = GhostConv(x1, 64, 3, strides=2)
    x2 = MultiScaleGatedAttention(x2)
    x2 = GELAN_Block(x2, 64, n=3)

    # Stage 3
    x3 = GhostConv(x2, 128, 3, strides=2)
    x3 = MultiScaleGatedAttention(x3)
    x3 = GELAN_Block(x3, 128, n=3)

    # Stage 4
    x4 = GhostConv(x3, 256, 3, strides=2)
    x4 = MultiScaleGatedAttention(x4)
    x4 = GELAN_Block(x4, 256, n=4)

    return x2, x3, x4

def ClassificationHead(x, num_classes):
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return outputs

# Full Model Builder
def A_MGA_GCY9_GELAN(input_shape=(32, 32, 3), num_classes=10, HN=None):
    inputs = Input(shape=input_shape)
    x2, x3, x4 = GCY9_GELAN_Backbone(inputs)
    # Fuse multiscale features
    x = Concatenate()([
        GlobalAveragePooling2D()(x2),
        GlobalAveragePooling2D()(x3),
        GlobalAveragePooling2D()(x4)
    ])
    x = Dense(HN, activation="relu")(x)
    x = Reshape((1, 1, 512))(x)
    x = ClassificationHead(x, num_classes)
    model = Model(inputs, x)
    return model

def Model_A_MGA_GCY9_GELAN(x_train, y_train, sol=None):
    if sol is None:
        sol =[5, 5]
    num_classes = len(np.unique(y_train))
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    model = A_MGA_GCY9_GELAN(input_shape=x_train.shape[1:],num_classes=num_classes, HN=int(sol[0]))
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    model.summary()
    model.fit(x_train, y_train_cat, epochs=int(sol[1]),batch_size=8,verbose=1)
    y_pred_prob = model.predict(x_train)
    y_pred = np.argmax(y_pred_prob, axis=1).reshape(-1, 1)
    y_true = y_train.reshape(-1, 1)
    evaluation = [net_evaluation(y_true[n].astype('uint8'),y_pred[n].astype('uint8'))for n in range(len(y_true))]
    return evaluation, y_pred


