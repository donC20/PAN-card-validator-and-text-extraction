from flask import Flask, jsonify, request
import cv2
import pytesseract
import numpy as np  
import tensorflow as tf 
import requests
import re
from io import BytesIO
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#augment
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = 'pytesseract/tesseract.exe'

application = Flask(__name__)

# Load the .h5 model
model = tf.keras.models.load_model("pan_detector_model_v7.h5")

def preprocess_image_from_url(image_url):
    response = requests.get(image_url)
    img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    # Apply augmentation
    # img = datagen.random_transform(img)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.float32(img)
    img = np.expand_dims(img, axis=0)
    return img
def extract_information(img):
    r = re.compile(r"([A-Z]+\s[A-Z]+\s[A-Z]+)|([A-Z]{5}[0-9]{4}[A-Z]{1})|([0-9]{2}/[0-9]{2}/[0-9]{4})")
    margin = 10
    # augmented_image_path = 'path_to_save_augmented_image.jpg'
    # cv2.imwrite(augmented_image_path, img)
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Apply thresholding
    img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create a mask to focus on left and center
    mask = img_thresh.copy()
    mask[:, int(mask.shape[1]*0.7):] = 0

    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    img_invert = 255 - img_opening
    data = pytesseract.image_to_data(img_gray, output_type='dict')
    bag = [(id, text) for id, text in enumerate(data['text']) if r.match(text)]
    img_marked = img.copy()

    for (id, text) in bag:
        if "Name" in text:
            name_region = img[data['top'][id]+data['height'][id]:data['top'][id]+2*data['height'][id], 
                          data['left'][id]:data['left'][id]+2*data['width'][id]]
    if 'name_region' in locals():
        name_gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)
        name_thresh = cv2.threshold(name_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        name_text = pytesseract.image_to_string(name_thresh)
        name_match = r.match(name_text)
        if name_match:
            bag.append((-1, name_text))
    extracted_info = {
        "Name": "",
        "Id": "",
        "Dob": ""
    }

    for (id, text) in bag:
        if id == -1:
            extracted_info["Name"] = text
        elif r.match(text):
            if re.match(r"[A-Z]{5}[0-9]{4}[A-Z]{1}", text):
                extracted_info["Id"] = text
            elif re.match(r"[0-9]{2}/[0-9]{2}/[0-9]{4}", text):
                extracted_info["Dob"] = text

    return extracted_info


def predictByImageFromURL(image_url):
    # Preprocess the input image
    input_image = preprocess_image_from_url(image_url)
    
    # Run the inference
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)
    class_labels = ["PAN", "Others"]
    
    predicted_class = class_labels[predicted_class_index]
    
    if predicted_class == "PAN":
        response = requests.get(image_url)
     
        img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

        # Apply augmentation
        # img = datagen.random_transform(img)
        # augmented_image_path = 'path_to_save_augmented_image.jpg'
        # cv2.imwrite(augmented_image_path, img)

        # Extract information using regular expressions
        info = extract_information(img)

        return jsonify({"predicted_class": predicted_class, "data": info})
    else:
        return jsonify({"predicted_class": "Not an PAN", "data": predicted_class})


@application.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        image_url = data.get('image_url')
        print("Received image_url:", image_url)
        return predictByImageFromURL(image_url)
    elif request.method == 'GET':
        image_url = request.args.get('image_url')
        print("Received image_url:", image_url)
        return predictByImageFromURL(image_url)
    else:
        return jsonify({"error": "Invalid request method"}), 405
@application.route('/')
def index():
    return "<center><h1>hello dude</h1></center>"


if __name__ == '__main__':
    application.run(debug=True)
