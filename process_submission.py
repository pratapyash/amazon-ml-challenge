import pandas as pd
from sanity import sanity_check

TEST_DATA_PATH = 'dataset/test.csv'
LENGTH = 131288
def fill_missing_indices(reference_file, csv_file, output_file, total_rows=LENGTH):
    # Read the reference CSV file (test.csv) into a DataFrame
    ref_df = pd.read_csv(reference_file)
    
    # Read the CSV file (test_out_1.csv) into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Ensure the 'index' column is sorted in both DataFrames
    ref_df = ref_df.sort_values(by='index').reset_index(drop=True)
    df = df.sort_values(by='index').reset_index(drop=True)
    
    # Create a DataFrame with the full index range from 0 to total_rows - 1
    full_index_range = range(total_rows)
    full_df = pd.DataFrame({'index': full_index_range})
    
    # Filter indices in full_df that are also present in the reference file but missing in the current file
    missing_indices = full_df[~full_df['index'].isin(df['index']) & full_df['index'].isin(ref_df['index'])]
    
    # Create a DataFrame for missing indices with empty 'prediction' values
    missing_df = pd.DataFrame({'index': missing_indices['index'], 'prediction': ''})
    
    # Concatenate the existing DataFrame with the missing indices DataFrame
    merged_df = pd.concat([df, missing_df]).sort_values(by='index').reset_index(drop=True)
    
    # Save the modified DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)
    print(f"Missing indices filled and saved to {output_file}")

# Example usage:
fill_missing_indices(TEST_DATA_PATH, 'test_out.csv', 'test_out.csv')
sanity_check(TEST_DATA_PATH, 'test_out.csv')