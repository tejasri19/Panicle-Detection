from flask import Flask, render_template, request, redirect, url_for
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from detect_panicles import detect_panicles
from kmeans import kmeans_panicle_detection

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    selected_method = request.form['method']

    # Read the uploaded image file
    img = Image.open(file)
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Detect panicles based on the selected method
    if selected_method == 'detect_panicles':
        result_image, num_panicles = detect_panicles(img_cv2)
    elif selected_method == 'kmeans_panicle_detection':
        result_image, num_panicles = kmeans_panicle_detection(img_cv2)
    else:
        # Handle the case if no method is selected
        return redirect(url_for('index'))

    # Convert the original image to base64 for embedding in HTML
    _, img_encoded = cv2.imencode('.jpg', img_cv2)
    original_image_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Convert the detected image to base64 for embedding in HTML
    _, result_encoded = cv2.imencode('.jpg', result_image)
    result_image_base64 = base64.b64encode(result_encoded).decode('utf-8')

    return render_template('result.html', input_image=original_image_base64, output_image=result_image_base64, count=num_panicles)


if __name__ == '__main__':
    app.run(debug=True, port=5578)
