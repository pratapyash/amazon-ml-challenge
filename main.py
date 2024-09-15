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
from entity_extractor import GeneralEntityExtractor, DimensionEntityExtractor


class PredictionProcessor:
    def __init__(self, test_df, test_images_dir):
        self.test_df = test_df
        self.test_images_dir = test_images_dir
        self.general_extractor = GeneralEntityExtractor()
        self.dimension_extractor = DimensionEntityExtractor()

    def process_sample(self, row):
        image_path = os.path.join(self.test_images_dir, os.path.basename(row['image_link']))
        entity_name = row['entity_name']
        
        if entity_name in ['depth', 'width', 'height']:
            extractor = self.dimension_extractor
        else:
            extractor = self.general_extractor
        
        extracted_text = extractor.process_image(image_path)
        predicted_value, predicted_unit = extractor.extract_entity_value(entity_name, extracted_text)
        formatted_prediction = extractor.format_prediction(predicted_value, predicted_unit)
        
        return {
            'index': row['index'],
            'prediction': formatted_prediction
        }

    def process_test_data(self, start_index=0, input_file=None, output_file='test_out.csv'):
        if input_file:
            input_predictions = pd.read_csv(input_file)
        else:
            input_predictions = pd.DataFrame(columns=['index', 'prediction'])

        filtered_test_df = self.test_df[self.test_df['index'] >= start_index]
        
        new_predictions = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_index = {executor.submit(self.process_sample, row): row['index'] for _, row in filtered_test_df.iterrows()}
            
            for future in tqdm(as_completed(future_to_index), total=len(filtered_test_df), desc="Processing test data"):
                index = future_to_index[future]
                try:
                    result = future.result()
                    if result is not None:
                        new_predictions.append(result)
                except Exception as exc:
                    print(f'Sample {index} generated an exception: {exc}')
                
                if len(new_predictions) % 100 == 0:
                    updated_predictions = self.update_predictions(input_predictions, new_predictions)
                    self.save_predictions(updated_predictions, output_file)
        
        final_predictions = self.update_predictions(input_predictions, new_predictions)
        self.save_predictions(final_predictions, output_file)
        return final_predictions

    @staticmethod
    def update_predictions(input_predictions, new_predictions):
        new_pred_dict = {pred['index']: pred['prediction'] for pred in new_predictions}
        input_predictions.loc[input_predictions['index'].isin(new_pred_dict.keys()), 'prediction'] = \
            input_predictions.loc[input_predictions['index'].isin(new_pred_dict.keys()), 'index'].map(new_pred_dict)
        return input_predictions

    @staticmethod
    def save_predictions(predictions, output_file):
        predictions.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

# Main execution
if __name__ == "__main__":
    # Load data
    test_df = pd.read_csv('dataset/test.csv')
    TEST_IMAGES_DIR = 'test_images'

    # Initialize and run prediction processor
    processor = PredictionProcessor(test_df, TEST_IMAGES_DIR)
    predictions = processor.process_test_data(start_index=0, input_file='test_out6.csv', output_file='test_out_final.csv')

    # Run sanity check
    from sanity import sanity_check
    sanity_check('dataset/test.csv', 'test_out_final.csv')

    print("Final predictions saved successfully.")