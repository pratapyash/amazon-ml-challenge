import pandas as pd
import re
import os
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from tqdm import tqdm
from constants import entity_unit_map
from patterns import entity_patterns, entity_keywords, unit_mappings
from sanity import sanity_check
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize OCR model
model_ocr = ocr_predictor(pretrained=True, assume_straight_pages=False, export_as_straight_boxes=True)

# Load the test data
test_df = pd.read_csv('dataset/test.csv')

# Set the path to your test images directory
TEST_IMAGES_DIR = 'test_images'

# Function to extract text from the OCR result
def extract_text(ocr_result):
    extracted_text = []
    for page in ocr_result['pages']:
        for block in page['blocks']:
            for line in block['lines']:
                for word in line['words']:
                    extracted_text.append(word['value'])
    return ' '.join(extracted_text)

# Function to process a single image
def process_image(image_path):
    try:
        doc = DocumentFile.from_images(image_path)
        result = model_ocr(doc)
        extracted_text = extract_text(result.export())
        return extracted_text
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return ""

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9.\s\'"]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Unit standardization
def standardize_unit(unit):
    unit = unit.lower()
    return unit_mappings.get(unit, unit)

def extract_value_unit(text, pattern, allowed_units):
    if not isinstance(pattern, str):
        return []
    matches = re.findall(pattern, text)
    extractions = []
    for match in matches:
        value = float(match[0])
        unit = match[2]
        unit_standard = standardize_unit(unit)
        if unit_standard in allowed_units:
            extractions.append((value, unit_standard))
    return extractions

def find_value_with_context(text, entity, pattern, allowed_units):
    if not isinstance(pattern, str):
        return []
    keywords = entity_keywords.get(entity, [])
    extractions = []
    for keyword in keywords:
        keyword_positions = [m.start() for m in re.finditer(keyword, text)]
        for pos in keyword_positions:
            window = text[max(0, pos - 50): pos + 50]
            extractions.extend(extract_value_unit(window, pattern, allowed_units))
    if not extractions:
        extractions = extract_value_unit(text, pattern, allowed_units)
    return extractions

def extract_entity_value(entity, text):
    clean_text = preprocess_text(text)
    allowed_units = entity_unit_map.get(entity, set())
    pattern = entity_patterns.get(entity, '')
    extractions = find_value_with_context(clean_text, entity, pattern, allowed_units)
    
    if extractions:
        if entity in ['width', 'height', 'depth']:
            if len(extractions) >= 2:
                sorted_extractions = sorted(extractions, key=lambda x: clean_text.find(f"{x[0]} {x[1]}"))
                
                if entity == 'width':
                    return sorted_extractions[0]
                elif entity == 'depth':
                    return sorted_extractions[1] if len(extractions) > 1 else sorted_extractions[0]
                elif entity == 'height':
                    return sorted_extractions[-1]
            else:
                return extractions[0]
        else:
            return extractions[0]
    
    return "", ""

# Function to format the prediction
def format_prediction(value, unit):
    if value and unit:
        return f"{value:.2f} {unit}"
    return ""

# Function to save predictions to CSV
def save_predictions(predictions, output_file):
    pd.DataFrame(predictions).to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# Function to process a single sample
def process_sample(row):
    image_path = os.path.join(TEST_IMAGES_DIR, os.path.basename(row['image_link']))
    extracted_text = process_image(image_path)
    entity_name = row['entity_name']
    
    predicted_value, predicted_unit = extract_entity_value(entity_name, extracted_text)
    formatted_prediction = format_prediction(predicted_value, predicted_unit)
    
    return {
        'index': row['index'],
        'prediction': formatted_prediction
    }

# Main function to process test data and generate predictions
def process_test_data():
    predictions = []
    output_file = 'test_out.csv'
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_index = {executor.submit(process_sample, row): index for index, row in test_df.head(100).iterrows()}
        
        for future in tqdm(as_completed(future_to_index), total=100, desc="Processing test data"):
            index = future_to_index[future]
            try:
                result = future.result()
                predictions.append(result)
            except Exception as exc:
                print(f'Sample {index} generated an exception: {exc}')
            
            # Save predictions every 100 samples
            if len(predictions) % 100 == 0:
                save_predictions(sorted(predictions, key=lambda x: x['index']), output_file)
    
    # Save final predictions
    save_predictions(sorted(predictions, key=lambda x: x['index']), output_file)
    return predictions

# Run the prediction process
predictions = process_test_data()

# Run sanity check
sanity_check('dataset/test.csv', 'test_out.csv')