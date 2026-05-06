import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from Classificaltion_Evaluation import net_evaluation
from tensorflow.keras import layers, models


def load_maskrcnn_model(num_classes):

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(256, 256, 3)
    )

    x = base_model.output

    # Feature Pyramid
    x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D((2,2))(x)

    # Mask Head
    mask_output = layers.Conv2D(num_classes, (1,1), activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=mask_output)

    return model


def Mask_RCNN(Data, Target):
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

    model = load_maskrcnn_model(NUM_CLASSES)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(Data,Target,epochs=10, batch_size=8,validation_split=0.2)
    pred = model.predict(Data)
    pred_labels = np.zeros_like(pred)
    pred_labels[np.arange(pred.shape[0]), np.argmax(pred, axis=1)] = 1
    Eval = [net_evaluation(Target[n].astype('uint8'), pred_labels[n].astype('uint8')) for n in range(pred.shape[0])]
    return Eval, pred



