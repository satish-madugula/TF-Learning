import tensorflow as tf
import numpy as np
from tensorflow import keras

# load fashion mnist from tesnorflow keras datasets

fmnist = tf.keras.datasets.fashion_mnist
(train_img, train_lbl),(test_img, test_lbl) = fmnist.load_data()