import numpy as np
from keras import utils 
from keras.applications.resnet import ResNet152, preprocess_input
from keras.applications.imagenet_utils import decode_predictions







#Now define a function that resizes the input image to 299x299 pxiels and then converts
#the image to a NumPy array
def preprocess_image(im):
    img = utils.load_img(im, target_size=(224, 224))
    img = utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img



def classify_image_using_tensorflow(imagepath):
    predicted_labels = []

    model = ResNet152(include_top=True, weights='imagenet')
    #model.summary()  # Uncomment this to print a long summary!

    preprocessed_image = preprocess_image(imagepath)
    pred = model.predict(preprocessed_image)

    for prediction in decode_predictions(pred)[0]:
        predicted_labels.append(prediction)
    return predicted_labels


if __name__=="__main__":
    labels = classify_image_using_tensorflow("../input_images/aeroplane.jpg")
    for l in labels:
        print(l)

