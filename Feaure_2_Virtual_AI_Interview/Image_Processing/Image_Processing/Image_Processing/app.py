import cv2
import numpy as np
import base64
import os
import pytesseract
import pandas as pd
from flask import Flask, jsonify, request, render_template
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Load Excel data
excel_data = pd.read_excel('data.xlsx')

# Function to compare images
def compare_images(image1, image2):
    # Convert images to grayscale
    img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Perform template matching or any other image comparison technique
    # Compare histograms
    hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])

    # Calculate correlation
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    # Define threshold for similarity
    threshold = 0.8

    if correlation >= threshold:
        return True
    else:
        return False

# Function to extract name from ID screenshot
def extract_name_from_id(id_screenshot):
    # Use OCR to extract text from the ID screenshot
    id_text = pytesseract.image_to_string(id_screenshot)
 
    name = id_text.split('\n')[0]
    return name

# Route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for processing screenshots
@app.route('/process_screenshot', methods=['POST'])
def process_screenshot():
    data = request.get_json()
    screenshot = data['screenshot'].split(',')[1]  # Remove data URL prefix

    # Check if 'id_screenshot' is in the files
    if 'id_screenshot' in request.files:
        id_screenshot = request.files['id_screenshot'].read()

        # Perform image comparison
        result = compare_images(cv2.imdecode(np.frombuffer(base64.b64decode(screenshot), np.uint8), cv2.IMREAD_COLOR), cv2.imdecode(np.frombuffer(id_screenshot, np.uint8), cv2.IMREAD_COLOR))

        # Extract name from ID screenshot
        name_from_id = extract_name_from_id(id_screenshot)

        # Compare name with Excel sheet
        if name_from_id in excel_data['Name'].values:
            name_comparison_result = "Match"
        else:
            name_comparison_result = "No Match"

        if result:
            message = f"Match found: Person in ID is in the video. Name comparison result: {name_comparison_result}"
        else:
            message = f"No match found: Person in ID is not in the video. Name comparison result: {name_comparison_result}"
    else:
        message = "No 'id_screenshot' found in request files."

    print(message)  # Print result in the terminal
    return jsonify({'message': message})

# Route for processing first screenshot
@app.route('/process_first_screenshot', methods=['POST'])
def process_first_screenshot():
    data = request.get_json()
    screenshot = data['screenshot'].split(',')[1]  # Remove data URL prefix

    # Print some debug information
    print("Processing first screenshot...")
    print("Screenshot length:", len(screenshot))

    # Save the first screenshot temporarily (for demonstration)
    with open('first_screenshot.jpg', 'wb') as f:
        f.write(base64.b64decode(screenshot))

    # For now, just return a message
    return jsonify({'message': 'First screenshot processed successfully'})

# Route for processing second screenshot
@app.route('/process_second_screenshot', methods=['POST'])
def process_second_screenshot():
    data = request.get_json()
    screenshot = data['screenshot'].split(',')[1]  # Remove data URL prefix

    # Print some debug information
    print("Processing second screenshot...")
    print("Screenshot length:", len(screenshot))

    # Save the second screenshot temporarily (for demonstration)
    with open('second_screenshot.jpg', 'wb') as f:
        f.write(base64.b64decode(screenshot))

    # Load the first screenshot (assumed to be saved as 'first_screenshot.jpg')
    first_screenshot = cv2.imread('first_screenshot.jpg')

    # Convert the second screenshot to OpenCV image
    second_screenshot = cv2.imdecode(np.frombuffer(base64.b64decode(screenshot), np.uint8), cv2.IMREAD_COLOR)

    # Perform image comparison
    result = compare_images(first_screenshot, second_screenshot)

    if result:
        message = "Match found: Faces in both screenshots match."
    else:
        message = "No match found: Faces in the two screenshots do not match."

    print(message)  # Print result in the terminal
    return jsonify({'message': message})

if __name__ == '__main__':
    app.run(debug=True)
