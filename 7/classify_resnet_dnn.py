import os
import numpy as np
import cv2

imagenet_classes_filepath = "../weights/7/ILSVRC2012.txt"
resnet152_weights_filepath = "../weights/7/resnet/resnet152.pb"
resnet152_shape = (224, 224)

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
    model = cv2.dnn.readNet(resnet152_weights_filepath)
    im = cv2.imread(imagepath)
    resized_image = cv2.resize(im, resnet152_shape)    

    image_blob = cv2.dnn.blobFromImage(resized_image, 1.0, resnet152_shape, [103.939, 116.779, 123.68]) 

    model.setInput(image_blob)
    predictions = model.forward()
    return decode_predictions(predictions,imagenet_class_names, 7)

if __name__=="__main__":
    labels_and_confidences = classify_image_using_opencvdnn("../input_images/aeroplane.jpg")
    for inx in labels_and_confidences:
        print(inx)

