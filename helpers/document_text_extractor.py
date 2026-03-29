import os
import fitz
import magic

import document_ocr


class DocumentTextExtractor:
    def __init__(self):
        self.ocr_extractor = document_ocr.DocumentOCRExtractor()

    def extract_text(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return "File not found."

        try:
            mime_type = magic.from_file(file_path, mime=True)
        except Exception as e:
            return f"MIME type detection failed: {str(e)}"

        if mime_type == 'text/plain':
            return self._extract_from_txt(file_path)

        elif mime_type == 'application/pdf':
            return self._extract_from_pdf(file_path)

        elif mime_type and mime_type.startswith('image/'):
            return self.ocr_extractor.extract_from_image(file_path)

        else:
            return f"Unsupported file format: {mime_type}"

    def _extract_from_txt(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            return f".txt extraction failed: {str(e)}"

    def _extract_from_pdf(self, file_path: str) -> str:
        try:
            document = fitz.open(file_path)
            extracted_text_blocks = []
            needs_ocr = False

            for page in document:
                page_text = page.get_text().strip()
                embedded_images = page.get_images()

                if len(page_text) < 50 and len(embedded_images) > 0:
                    needs_ocr = True
                    break

            if needs_ocr:
                return self.ocr_extractor.extract_from_pdf(file_path)

            for page in document:
                extracted_text_blocks.append(page.get_text())

            extracted_text = "\n".join(extracted_text_blocks)

            return extracted_text

        except Exception as e:
            return f"PDF Extraction failed: {str(e)}"