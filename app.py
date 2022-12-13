from flask import Flask, request, jsonify
from PIL import Image, ImageFilter
from rembg import remove
import io
import tensorflow as tf
from tensorflow import keras
import numpy as np
import traceback
import json

from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello!'

@app.route('/transform', methods=['POST'])
def transform_image():
    
    if model:
        try:
            # Get the image data from the request
            image_data = request.get_data()
            
            transformed_image = transform_image(image_data) # Transform the image
            classification = classify_image(transformed_image) # Classify the image
            
            return jsonify({'Prediction': str(classification)})
        except:
            return jsonify({'Trace': traceback.format_exc()}) 
    else:
        return ('No model here to use!')
    
    
def transform_image(image_data):   
    # Open the image using PIL
    image = Image.open(io.BytesIO(image_data))
    
    # Remove the background from the image
    image = remove(image)

    # Add white background to the image
    new_image = Image.new('RGBA', image.size, 'WHITE')
    new_image.paste(image, (0, 0), image)              
    new_image = new_image.resize((299, 299))
    new_image = new_image.filter(ImageFilter.GaussianBlur(radius=1))
    new_image = new_image.convert('RGB')
    
    return new_image
    
def classify_image(img):
    img = image.load_img(img, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    prediction = model.predict(img_preprocessed)

    score = np.argmax(prediction)
    klass = [k for k, v in class_dict.items() if v == np.argmax(score)][0]
    
    return klass

if __name__ == '__main__':
    model = keras.models.load_model('../DASC2-challenge3/models/inceptionv3-4') # Load the model
    print ('Model loaded')
    
    with open('class_names.json') as json_file:
        class_dict = json.load(json_file)
    print('Class names loaded')
    
    app.run()
