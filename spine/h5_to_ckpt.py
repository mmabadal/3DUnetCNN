import tensorflow as tf
import matplotlib
from keras import backend as K

matplotlib.use('agg')


'''
script to 

inputs:
 - 

outputs:
 - 

execution example:
 - python3 f

'''


def main():

    #saver = tf.train.Saver()
    #model = K.models.load_model("/disk/spines_3d/3DUnetCNN/spine/results/spines_dendrite/test_save/spine_segmentation_model.h5")
    #sess = K.get_session()
    #save_path = saver.save(sess, "/disk/spines_3d/3DUnetCNN/spine/results/spines_dendrite/test_save/ckpt/model.ckpt")

    saver = tf.train.Checkpoint()
    model = K.models.load_model("/disk/spines_3d/3DUnetCNN/spine/results/spines_dendrite/test_save/spine_segmentation_model.h5", compile=False)
    sess = tf.compat.v1.keras.backend.get_session()
    save_path = saver.save("/disk/spines_3d/3DUnetCNN/spine/results/spines_dendrite/test_save/ckpt/model.ckpt")



if __name__ == "__main__":
    main()
