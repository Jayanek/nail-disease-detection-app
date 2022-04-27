import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sklearn
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.models import load_model
from PIL import Image
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.preprocessing import PowerTransformer
from sklearn import preprocessing
import joblib
import os
os.environ["TFHUB_CACHE_DIR"] = "models"


nail_types=['terry\'s nail','beau\'s lines','splinter hemmorrage','yellow nails','healthy nails']

nail_shapes=['clubbing','koilonychia','healthy_nails']

heart_disease_columns = ['Pregnancies','Glucose','BloodPressure','Insulin','BMI','Age']

heart_disease_outcomes = ['Having heart disease','Not having heart disease']

def load_the_nail_type_model():
    model = load_model('./models/model_l48_a838.h5')
    print("Model nail type loaded...")
    return model

def load_the_nail_shape_model():
    model = load_model('./models/mobile_net_updated.h5', custom_objects={'KerasLayer':hub.KerasLayer})
    print("Model nail shape loaded...")
    return model

def load_heart_disease_confirm_model():
    model = pickle.load(open('./models/reg_log.sav', 'rb'))
    power =  joblib.load('./preprocess/powerX.save')
    scaler =  joblib.load('./preprocess/min_maxscaler.save')  
    print("Model heart disease confirm loaded...")
    return (model,power,scaler)

def confirm_hear_disease(pregnancies, glucose,bloodPressure,insulin,bmi,age):
    (model,power,scaler) = load_heart_disease_confirm_model()
    X_power=power.transform([[pregnancies, glucose,bloodPressure,insulin,bmi,age]])
    print(X_power)
    x_scaled = scaler.transform(X_power)
    y_pred = model.predict(x_scaled)
    return y_pred


nail_type_model = load_the_nail_type_model()
nail_shape_model = load_the_nail_shape_model()
confirm_hear_disease(2,124,68,205,32.9,30)

def image_preprocess(image: Image.Image)-> Image.Image:
	image = np.asarray(image.resize((224, 224)))[..., :3]
	image = np.expand_dims(image, 0)
	#image = image / 255.0
	return image


def predict_nail_type(image: Image.Image):
    pre_processed_image = image_preprocess(image)
    result = nail_type_model.predict(pre_processed_image)
    
	# get the softmax probabilities
    score = tf.nn.softmax(result)
    print(score)
    # select max probability class
    response = np.argmax(score)
    return nail_types[response]


def predict_nail_shape(image: Image.Image):
    # pre-processed image normalize by / 255
    pre_processed_image = image_preprocess(image)/255
    result = nail_shape_model.predict(pre_processed_image)
	# get the softmax probabilities
    print(result)
    score = tf.nn.softmax(result)
    print(score)
    # select max probability class
    response = np.argmax(score)
    return nail_shapes[response]


