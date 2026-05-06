import numpy as np
import keras
from keras.models import Model
from keras.layers import (Input, Conv2D, BatchNormalization, Activation,
                          GlobalAveragePooling2D, Dense, concatenate,
                          Multiply, Reshape)
from keras.optimizers import Adam
from Classificaltion_Evaluation import net_evaluation


# Multi-Scale Convolution Block
def multi_scale_block(x, filters):
    conv3 = Conv2D(filters, (3, 3), padding='same')(x)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    conv5 = Conv2D(filters, (5, 5), padding='same')(x)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    conv7 = Conv2D(filters, (7, 7), padding='same')(x)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    out = concatenate([conv3, conv5, conv7])
    return out

# Channel Attention Block
def attention_block(x, filters):
    gap = GlobalAveragePooling2D()(x)
    dense1 = Dense(filters // 4, activation='relu')(gap)
    dense2 = Dense(filters, activation='sigmoid')(dense1)
    scale = Reshape((1, 1, filters))(dense2)
    out = Multiply()([x, scale])
    return out

# DMSAU Network Architecture
def dmsau_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = multi_scale_block(inputs, 32)
    x = multi_scale_block(x, 64)

    x = attention_block(x, x.shape[-1])

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

def Model_DMSAU_Net(x_train, y_train):
    num_classes = len(np.unique(y_train))
    input_shape = x_train.shape[1:]
    y_train = y_train.astype(int).flatten()
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    model = dmsau_model(input_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    model.fit(x_train,y_train_cat, epochs=20, batch_size=8, verbose=1)
    predictions = model.predict(x_train)
    pred_labels = np.argmax(predictions, axis=1)
    y_true = y_train.reshape(-1, 1)
    y_pred = pred_labels.reshape(-1, 1)
    evaluation = [net_evaluation(y_true[n].astype('uint8'),y_pred[n].astype('uint8'))for n in range(len(y_true))]
    evaluation = np.mean(evaluation, axis=0)
    return evaluation, predictions

