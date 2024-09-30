# Hindi English OCR with Keyword Search using Qwen2-VL

![{60950997-B613-4DCA-A2F0-AF993237BDE1}](https://github.com/user-attachments/assets/3c742784-509b-4198-8e91-75d19b519f9f)

This project implements a powerful OCR tool that utilizes the **Qwen2-VL** model for extracting text from images in Hindi and English. Users can upload images, extract the text, and optionally search for specific keywords within the extracted text.

## Features

- Upload images with text in Hindi or English.
- Extracted text displayed in a user-friendly interface.
- Optional keyword search functionality to highlight occurrences in the extracted text.

## Requirements

Before running the application, ensure you have the following Python packages installed:

```bash
pip install gradio transformers Pillow torch

```
Usage
Clone the repository:
```
git clone https://github.com/Ashutosh0x/OCR-and-Keyword-Search-with-Qwen2-VL.git
cd OCR-and-Keyword-Search-with-Qwen2-VL 
```

Install the required packages.

Run the application:
python app.py

How It Works
The application uses the Qwen2-VL model to perform OCR on uploaded images.
Users can select a language for the OCR process and enter keywords for searching within the extracted text.
The app provides instant feedback and displays both the extracted text and search results.
Model
This application utilizes the Qwen2-VL model, which is designed for visual-language tasks, enabling efficient text extraction and processing from images.

Gradio for the easy-to-use interface.
Transformers library for model handling.
Pillow for image processing.

Deploying on Hugging Face
You can easily deploy this application on Hugging Face Spaces. Follow these steps:

Go to Hugging Face Spaces.
Create a new Space and choose the "Gradio" option.
Upload your app.py file and any other necessary files (like requirements.txt).
Once uploaded, Hugging Face will automatically build and run your application.
