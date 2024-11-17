#!/usr/bin/env bash

# Update the package list
apt-get update 

# Install Tesseract OCR
apt-get install -y tesseract-ocr

# Install any additional languages for Tesseract (optional)
# Uncomment the line below if you want to support additional languages like French, Spanish, etc.
# apt-get install -y tesseract-ocr-fra tesseract-ocr-spa

# Verify Tesseract installation
tesseract --version
