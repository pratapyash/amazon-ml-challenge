import pandas as pd
import re
import os
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod

# Constants and mappings
from constants import entity_unit_map
from patterns import entity_patterns, entity_keywords, unit_mappings

class EntityExtractor(ABC):
    def __init__(self):
        self.model_ocr = ocr_predictor(pretrained=True, assume_straight_pages=False, export_as_straight_boxes=True).cuda()

    @abstractmethod
    def extract_entity_value(self, entity, text):
        pass

    def process_image(self, image_path):
        try:
            doc = DocumentFile.from_images(image_path)
            result = self.model_ocr(doc)
            extracted_text = self.extract_text(result.export())
            return extracted_text
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return ""

    @staticmethod
    def extract_text(ocr_result):
        extracted_text = []
        for page in ocr_result['pages']:
            for block in page['blocks']:
                for line in block['lines']:
                    for word in line['words']:
                        extracted_text.append(word['value'])
        return ' '.join(extracted_text)

    @staticmethod
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9.\s\'"]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def standardize_unit(unit):
        unit = unit.lower()
        return unit_mappings.get(unit, unit)

    @staticmethod
    def format_prediction(value, unit):
        if value and unit:
            return f"{value} {unit}"
        return ""
    
class DimensionEntityExtractor(EntityExtractor):
    def __init__(self):
        super().__init__()
        self.unit_factors = {
            'mm': 0.1, 'millimeter': 0.1, 'millimetre': 0.1,
            'cm': 1.0, 'centimeter': 1.0, 'centimetre': 1.0,
            'm': 100.0, 'meter': 100.0, 'metre': 100.0,
            'in': 2.54, 'inch': 2.54,
            'ft': 30.48, 'foot': 30.48, 'feet': 30.48,
        }

    def extract_entity_value(self, entity, text):
        clean_text = self.preprocess_text(text)
        dimensions = self.extract_dimensions(clean_text)
        dimensions += self.extract_range_dimensions(clean_text)

        if not dimensions:
            return "", ""

        if entity == 'depth':
            target_dimension = max(dimensions, key=lambda x: x['value_cm'])
        elif entity in ['width', 'height']:
            target_dimension = min(dimensions, key=lambda x: x['value_cm'])
        else:
            target_dimension = dimensions[0]

        value, unit = re.match(r'(\d+(?:\.\d+)?)(.*)', target_dimension['original']).groups()
        value = float(value)
        unit = self.standardize_unit(unit)
        return value, unit

    def extract_dimensions(self, text):
        pattern = r'(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mm|millimeter|millimetre|cm|centimeter|centimetre|m|meter|metre|in|inch|ft|foot|feet|yd|yard|"|\'|centimetres|millimetres|metres|inches|feet|yards)s?'
        matches = re.finditer(pattern, text, re.IGNORECASE)
        dimensions = []
        for match in matches:
            value = float(match.group('value'))
            unit = match.group('unit').lower()
            factor = self.unit_factors.get(unit, None)
            if factor:
                value_cm = value * factor
                dimensions.append({'original': f"{value}{unit}", 'value_cm': value_cm, 'position': match.start()})
        return dimensions

    def extract_range_dimensions(self, text):
        range_pattern = r'(?P<start>\d+\.?\d*)\s*(?P<unit>mm|millimeter|millimetre|cm|centimeter|centimetre|m|meter|metre|in|inch|ft|foot|feet|yd|yard|"|\'|centimetres|millimetres|metres|inches|feet|yards)s?\s*-\s*(?P<end>\d+\.?\d*)\s*(?P=unit)'
        range_matches = re.finditer(range_pattern, text, re.IGNORECASE)
        dimensions = []
        for match in range_matches:
            start_value = float(match.group('start'))
            end_value = float(match.group('end'))
            unit = match.group('unit').lower()
            factor = self.unit_factors.get(unit, None)
            if factor:
                start_cm = start_value * factor
                end_cm = end_value * factor
                dimensions.append({'original': f"{start_value}{unit}", 'value_cm': start_cm, 'position': match.start()})
                dimensions.append({'original': f"{end_value}{unit}", 'value_cm': end_cm, 'position': match.start()})
        return dimensions