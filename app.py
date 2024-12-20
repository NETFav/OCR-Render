from flask import Flask, request, jsonify
import pytesseract
from PIL import Image
import cv2
import numpy as np
import os
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Use environment variable or set the default Linux path
pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH', '/usr/bin/tesseract')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            # Improved preprocessing and OCR processing
            processed_image_path, processed_image_no_boxes_path = improved_preprocess_image(file_path)

            # Use pytesseract to extract text from the image without bounding boxes
            processed_image_no_boxes = Image.open(processed_image_no_boxes_path)
            extracted_text = pytesseract.image_to_string(processed_image_no_boxes)

            # Classify document based on the extracted text
            document_type = classify_document(extracted_text)

            # Return JSON with the document type and extracted text
            return jsonify({"document_type": document_type, "extracted_text": extracted_text})

        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500

def improved_preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise the image
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(denoised_image, 50, 150)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        # Process lines if needed
        processed_image = denoised_image
    else:
        # If no lines are detected, use the original image
        processed_image = denoised_image

    # Apply Otsu's thresholding after denoising
    _, otsu_thresh_image = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use adaptive thresholding for fine detail extraction
    adaptive_thresh_image = cv2.adaptiveThreshold(otsu_thresh_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 11, 2)

    # Apply morphological operations to clean up noise and isolate text
    kernel = np.ones((2, 2), np.uint8)
    morph_image = cv2.morphologyEx(adaptive_thresh_image, cv2.MORPH_CLOSE, kernel)

    # Save the processed image without bounding boxes for text extraction
    processed_image_no_boxes_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image_no_boxes.png')
    cv2.imwrite(processed_image_no_boxes_path, morph_image)

    # Convert the image to BGR format to draw colored bounding boxes
    morph_image_bgr = cv2.cvtColor(morph_image, cv2.COLOR_GRAY2BGR)

    # Detect words and draw bounding boxes
    d = pytesseract.image_to_data(morph_image, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(morph_image_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red color (BGR format)

    # Save the processed image with bounding boxes for visualization
    processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image_with_boxes.png')
    cv2.imwrite(processed_image_path, morph_image_bgr)

    return processed_image_path, processed_image_no_boxes_path

def classify_document(text):
    # Convert text to lowercase for easier comparison
    text_lower = text.lower()

    # Define patterns for contextual matching
    patterns = {
        "Birth Certificate": [r"\bbirth certificate\b", r"\bcertificate of live birth\b"],
        "Enrollment Form": [r"\benrollment form\b", r"\bbasic education\b"],
        "Report Card": [r"\bform-137\b", r"\bpermanent record\b", r"\breport card\b", 
        r"\blearning areas\b", r"\bdepartment of education\b", r"\blearning progress\b", r"\breport\b"]
    }

    # Dictionary to store count of matches for each type
    match_counts = {
        "Birth Certificate": 0,
        "Enrollment Form": 0,
        "Report Card": 0
    }

    # Check for patterns in the text
    for document_type, regex_list in patterns.items():
        for regex in regex_list:
            if re.search(regex, text_lower):
                match_counts[document_type] += 1

    # Determine the document type based on the highest count of keyword matches
    detected_type = max(match_counts, key=match_counts.get)

    # Additional check if no keywords were detected (handling 2x2 ID Picture)
    if sum(match_counts.values()) == 0 and len(text.strip()) == 0:
        return "1x1 ID Picture"
    elif sum(match_counts.values()) == 0:
        return "Unknown Document"
    else:
        return detected_type

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
