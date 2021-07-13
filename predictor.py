import os

import numpy as np

from keras.preprocessing import image
from extract_bottleneck_features import extract_Resnet50

CURRENT_DIR = os.path.dirname(__file__)


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def get_dog_names(input_file):
    with open(f'data/{input_file}') as f:
        lines = f.readlines()
        dog_names = list(lines[0].replace('\n', '').split(','))
    return dog_names


def predict_breed(model, img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    dog_names = get_dog_names('dog_names.csv')
    breed = dog_names[np.argmax(predicted_vector)].split('.')[-1]
    return breed
