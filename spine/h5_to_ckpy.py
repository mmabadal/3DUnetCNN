
from keras import backend as K
from keras.models import load_model
import keras.losses
import tensorflow as tf


model = load_model("spine_segmentation_model.h5",compile=False)
session = K.get_session()

saver = tf.train.Saver()
saver.save(session, "ckpt/test.ckpt")


