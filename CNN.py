#Authors: Advait Balaji (@advaitb) and Dharm Skandh Jain

import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.constraints import maxnorm
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.convolutional import UpSampling2D
import cv2
from keras import backend as K
import json
from keras.models import model_from_json
from skimage.transform import resize

def IoUscore(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    rounded = K.round(y_pred)
    y_pred_f = K.flatten(rounded)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f)

    return intersection/(union - intersection)

if __name__ == "__main__":
        # obj = Segment('dataset/')
        # obj.train()
        # obj.save_model(name="segment.gz")

        obj = Segment()
        obj.train()
        obj.save_model()

        list_id = [i for i in range(5)]
        train = []
        for id in list_id:
            input_file = 'Train_Data/valid-'+str(id)+'.jpg'
            input_img = cv2.imread(input_file)/255.0
            train.append(input_img)

        valid_set = train

        for i in range(0,5):
            predicted = obj.get_mask(valid_set[i])
            # print valid_set[i].shape , predicted.shape
            # predicted = predicted.reshape((predicted.shape[1],predicted.shape[2]))
            file_name = 'valid'+str(i)+'new.jpg'
            cv2.imwrite(file_name, np.array(predicted))
