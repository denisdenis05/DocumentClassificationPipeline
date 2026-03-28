import document_date_extractor
import document_ocr

document_OCR = document_ocr.DocumentOCRExtractor()
date_extractor = document_date_extractor.DocumentDateExtractor()

document_text = document_OCR.extract_from_pdf('1515brewing_20231217_010.pdf')
print(document_text)
print(f"result: {date_extractor.extract_primary_date(document_text)}")