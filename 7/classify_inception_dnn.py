import os
import numpy as np
import cv2

imagenet_classes_filepath = "../weights/7/ILSVRC2012.txt"
inceptionv3_weights_filepath = "../weights/7/inceptionv3/inceptionv3.pb"
inceptionv3_shape = (299,299)

def decode_predictions(predictions, class_names, top=5):
    results = []
    top_indices = predictions[0].argsort()[-top:][::-1]
    for i in top_indices:
        result = class_names[i] +": "+ str(predictions[0][i])
        results.append(result)
    return results

def classify_image_using_opencvdnn(imagepath):
    imagenet_class_names = None
    with open(imagenet_classes_filepath, 'rt') as f:
        imagenet_class_names = f.read().rstrip('\n').split('\n')

    # Load the model from disk
    model = cv2.dnn.readNet(inceptionv3_weights_filepath)
    im = cv2.imread(imagepath)
    resized_image = cv2.resize(im, inceptionv3_shape)

    image_blob = cv2.dnn.blobFromImage(resized_image, 1/127.5, inceptionv3_shape, [127.5, 127.5, 127.5]) 

    model.setInput(image_blob)
    predictions = model.forward()
    return decode_predictions(predictions,imagenet_class_names, 7)


if __name__=="__main__":
    labels_and_confidences = classify_image_using_opencvdnn("../input_images/aeroplane.jpg")
    for inx in labels_and_confidences:
        print(inx)

