import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from tensorflow.keras.models import load_model
from PIL import Image


target_names=['aloperia areata','beau\'s lines','Darier\'s disease','Muehrck-e\'s lines']

def load_the_model():
    model = load_model('basic_model.h5',custom_objects={'KerasLayer':hub.KerasLayer})
    print("Model loaded")
    return model

model = load_the_model()


def image_preprocess(image: Image.Image)-> Image.Image:
	image = np.asarray(image.resize((224, 224)))[..., :3]
	image = np.expand_dims(image, 0)
	image = image / 255.0
	return image


def predict(image: Image.Image):
    pre_processed_image = image_preprocess(image)
    result = model.predict(pre_processed_image)
	# get the softmax probabilities
    score = tf.nn.softmax(result)
    # select max probability class
    response = np.argmax(score)
    return target_names[response]