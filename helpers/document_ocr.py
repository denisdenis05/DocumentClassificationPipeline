import pytesseract
from pdf2image import convert_from_path
import os
from helpers import image_preprocessor


class DocumentOCRExtractor:
    def __init__(self):
        self.preprocessor = image_preprocessor.ImagePreprocessor()

    def extract_from_pdf(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return "File not found."

        try:
            pages = convert_from_path(file_path)
            extracted_text_blocks = []

            for page in pages:
                text = self.extract_from_image(page)
                extracted_text_blocks.append(text)

            extracted_text = "\n".join(extracted_text_blocks)

            return extracted_text

        except Exception as e:
            return f"OCR Extraction failed: {str(e)}"

    def extract_from_image(self, image) -> str:
        try:
            preprocessed_page = self.preprocessor.process_for_ocr(image)

            return pytesseract.image_to_string(preprocessed_page)
        except Exception as e:
            return f"OCR Extraction failed: {str(e)}"
