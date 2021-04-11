import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import QuoridorState

number_of_input_side = 100
number_of_feature_vector = 1024
#number_of_feature_vector = 512
# number_of_feature_vector = number_of_input_side * number_of_input_side
number_of_actions = 140
kernel_length = 5


def create_feature_vector_and_value_func_model():
    inputs = tf.keras.Input(shape=(number_of_input_side, number_of_input_side, 1))
    conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[kernel_length, kernel_length], padding='SAME', activation=tf.nn.relu, use_bias=False)(inputs)
    pool1 = tf.keras.layers.MaxPool2D(padding='SAME')(conv1)
    conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[kernel_length, kernel_length], padding='SAME', activation=tf.nn.relu, use_bias=False)(pool1)
    pool2 = tf.keras.layers.MaxPool2D(padding='SAME')(conv2)
    conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=[kernel_length, kernel_length], padding='SAME', activation=tf.nn.relu, use_bias=False)(pool2)
    pool3 = tf.keras.layers.MaxPool2D(padding='SAME')(conv3)
    pool3_flat = tf.keras.layers.Flatten()(pool3)

    feature_vector = tf.keras.layers.Dense(units=number_of_feature_vector, activation=tf.nn.relu, use_bias=False)(pool3_flat)

    value_func = tf.keras.layers.Dense(units=1, use_bias=False)(feature_vector)
    # logits = tf.keras.layers.Dense(units=140, activation=tf.nn.relu)(feature_vector)

    return tf.keras.Model(inputs=inputs, outputs=[feature_vector, value_func])


# sigmoid를 relu로 바꾸니 이동에 해당하는 logit값이 큰 폭으로 증가
def create_logit_model():
    # inputs = tf.keras.Input(shape=(number_of_input_side, 1))
    inputs = tf.keras.Input(shape=(number_of_feature_vector, 1))
    inputs_flat = tf.keras.layers.Flatten()(inputs)

    logits = tf.keras.layers.Dense(units=number_of_actions, activation=tf.nn.relu, use_bias=False)(inputs_flat)
    # logits = tf.keras.layers.Dense(units=140)(inputs_flat)

    return tf.keras.Model(inputs=inputs, outputs=logits)


def direct_value_func_model():
    inputs = tf.keras.Input(shape=(number_of_input_side * number_of_input_side, 1))
    pool3_flat = tf.keras.layers.Flatten()(inputs)

    value_func = tf.keras.layers.Dense(units=1, use_bias=False)(pool3_flat)

    return tf.keras.Model(inputs=inputs, outputs=value_func)