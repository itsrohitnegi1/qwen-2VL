import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import re

# Load the Qwen2-VL model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map={"": "cpu"}
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Function to perform OCR using Qwen2-VL
def ocr_image(image, language):
    try:
        # Prepare the input format for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract the text from this image."},
                ],
            }
        ]

        # Process the input for the model
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=image, text=[text_input], padding=True, return_tensors="pt").to("cpu")

        # Generate text using the model
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        return extracted_text
    except Exception as e:
        return f"Error occurred during OCR: {str(e)}"

# Function to perform keyword search within the extracted text
def search_text(text, keyword):
    if not text:
        return "No text extracted."

    # Search for keyword occurrences (case insensitive)
    keyword_pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    matches = keyword_pattern.findall(text)

    if matches:
        # Highlight all matches in the text by wrapping them with ** for bold
        highlighted_text = re.sub(keyword_pattern, f"**{keyword}**", text)
        return highlighted_text
    else:
        return "Keyword not found in the text."

# Function to handle both OCR and search functionality
def process_image(image, language, keyword=""):
    extracted_text = ocr_image(image, language)

    if keyword:
        result_text = search_text(extracted_text, keyword)
    else:
        result_text = extracted_text

    return extracted_text, result_text

# Gradio Interface
def build_interface():
    with gr.Blocks() as interface:
        gr.Markdown("## Hindi & English OCR with Keyword Search using Qwen2-VL")
        gr.Markdown("Upload an image with text in Hindi or English, extract the text, and optionally search for keywords within it.")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload an Image", elem_id="image-input", height=400)

            with gr.Column(scale=1):
                # Dropdown for selecting language (currently not used by Qwen2-VL, but kept for future expansion)
                language_input = gr.Dropdown(label="Select Language for OCR", choices=["English", "Hindi", "Both"], value="Both", elem_id="lang-dropdown", interactive=True)

                keyword_input = gr.Textbox(label="Enter Keyword to Search (Optional)", value="", placeholder="Enter keyword here", elem_id="keyword-input", interactive=True)

        with gr.Row():
            text_output = gr.Textbox(label="Extracted Text", interactive=False, placeholder="Extracted text will appear here.", elem_id="text-output", lines=10)
            search_output = gr.Textbox(label="Search Results", interactive=False, placeholder="Search results will appear here.", elem_id="search-output", lines=10)

        # Button to submit the image, language, and keyword
        submit_btn = gr.Button("Process", elem_id="submit-btn")

        # Markdown for feedback during processing
        processing_message = gr.Markdown("Processing...", visible=False, elem_id="processing-message")

        def on_submit(image, language, keyword):
            if not image:
                return "No image provided", "Please upload an image.", gr.update(visible=False)

            # Display the processing message
            extracted_text, result_text = process_image(image, language, keyword)
            return extracted_text, result_text, gr.update(visible=False)

        # Connect the button to the function with inputs and outputs
        submit_btn.click(on_submit,
                         inputs=[image_input, language_input, keyword_input],
                         outputs=[text_output, search_output, processing_message])

    return interface

# Main function to launch the app
if __name__ == "__main__":
    interface = build_interface()
    interface.launch(share=True, inbrowser=True, debug=True, height=900, width=1600)
