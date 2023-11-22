import tensorflow as tf
import functools


def ANN():
    input_shape = [900]
    inputt = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Dense(400)(inputt)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(5)(x)
    x = tf.keras.layers.Softmax()(x)
    m = tf.keras.Model(inputt,x)
    return m

