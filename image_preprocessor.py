import uuid

import cv2
import numpy as np
import os

from PIL import Image


class ImagePreprocessor:
    def process_for_ocr(self, image_input) -> np.ndarray:

        image = self.get_image_array(image_input)
        image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 3)

        binary = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        inverted_binary = cv2.bitwise_not(binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated_image = cv2.dilate(inverted_binary, kernel, iterations=1)

        final_processed_image = cv2.bitwise_not(dilated_image)

        return final_processed_image

    def process_image_file_for_ocr(self, file_path: str) -> str:
        processed_image_binary = self.process_for_ocr(file_path);

        image_file = f"processed_output{uuid.uuid4()}.jpg"
        cv2.imwrite(image_file, processed_image_binary)

        return image_file

    def get_image_array(self, image_input):
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at path: {image_input}")
            image = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise TypeError("Invalid image input type for preprocessing")

        return image