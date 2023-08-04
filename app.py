import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the model
model = tf.keras.models.load_model('path_to_saved_model')

app = Flask(__name__)

@app.route('/classify_garbage', methods=['POST'])
def classify_garbage():
    # Get the image data from the request
    image = request.files['image'].read()
    image = Image.open(io.BytesIO(image))
    
    # Preprocess the image for the model
    image = image.resize((224, 224))  # Assuming the model expects 224x224 images
    image = img_to_array(image)
    image = preprocess_input(image)
    image = tf.expand_dims(image, axis=0)  # Add a batch dimension

    # Make predictions using the model
    prediction = model.predict(image)
    
    # Assuming you have a list of class labels to interpret the prediction
    class_labels = ['class1', 'class2', 'class3']  # Replace with actual class labels
    predicted_class = class_labels[tf.argmax(prediction, axis=1)[0]]

    return jsonify({'result': predicted_class})

if __name__ == '__main__':
    app.run()


