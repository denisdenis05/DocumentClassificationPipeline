import os

import document_date_extractor
import document_classifier
import document_text_extractor

document_classifier = document_classifier.DocumentClassifier()
text_extractor = document_text_extractor.DocumentTextExtractor()
date_extractor = document_date_extractor.DocumentDateExtractor()


def document_classification(full_path: str) -> tuple:
    text = text_extractor.extract_text(full_path)
    date = date_extractor.extract_primary_date(text)
    classification = document_classifier.classify_text(text)
    return text, date, classification

for root, directories, files in os.walk('dataset'):
    for file_name in files:
        full_path_string = os.path.join(root, file_name)

        text, date, classification = document_classification(full_path_string)

        target_directory = os.path.join("result", classification, date)
        os.makedirs(target_directory, exist_ok=True)
        target_path_string = os.path.join(target_directory, file_name)

        with open(full_path_string, 'rb') as source_file:
            with open(target_path_string, 'wb+') as dest_file:
                dest_file.write(source_file.read())
