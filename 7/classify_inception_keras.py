import os
import numpy as np
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image as image_utils
from keras import utils  


def preprocess_image(im):
    img = utils.load_img(im, target_size=(299, 299))
    img = utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img




def classify_image_using_tensorflow(imagepath):
    predicted_labels = []

    model = InceptionV3(include_top=True, weights='imagenet')
    preprocessed_image = preprocess_image(imagepath)
    pred = model.predict(preprocessed_image)
    for prediction in decode_predictions(pred, top=7)[0]:
        predicted_labels.append(prediction)
    return predicted_labels



if __name__=="__main__":
    labels = classify_image_using_tensorflow("../input_images/aeroplane.jpg")
    for l in labels:
        print(l)

