import datefinder
from typing import Optional, List
import datetime

class DocumentDateExtractor:
    def __init__(self, require_strict_parsing: bool = True, target_date_format: str = "%Y-%m-%d"):
        self.require_strict_parsing = require_strict_parsing
        self.target_date_format = target_date_format

    def extract_primary_date(self, document_content: str) -> Optional[str]:
        extracted_date_objects: List[datetime.datetime] = list(
            datefinder.find_dates(document_content, strict=self.require_strict_parsing)
        )

        if not extracted_date_objects:
            return "Unknown"

        primary_date_object = extracted_date_objects[0]
        standardized_date_string = primary_date_object.strftime(self.target_date_format)

        return standardized_date_string
