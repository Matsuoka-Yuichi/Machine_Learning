import pandas as pd
import numpy as np

def process_and_reshape_csv(input_csv_path, output_csv_path):
    # Read the CSV, assuming the first row contains headers
    df = pd.read_csv(input_csv_path, usecols=[0, 1, 2])

    # Number of rows per group
    rows_per_group = 38

    # Create a new DataFrame to store the reshaped data
    reshaped_data = pd.DataFrame()

    # Process each group
    for n in range((df.shape[0] + rows_per_group - 1) // rows_per_group):
        start_row = n * rows_per_group
        end_row = min(start_row + rows_per_group, df.shape[0])
        
        # Extract the group for columns 2 and 3
        group = df.iloc[start_row:end_row, 1:3].reset_index(drop=True)
        
        # Rename columns to indicate the group
        group.columns = [f'Group {n+1} - Column 2', f'Group {n+1} - Column 3']
        
        # Append the group data as new columns in the reshaped DataFrame
        reshaped_data = pd.concat([reshaped_data, group], axis=1)
    
    # Fill any missing values with NaN or another placeholder if necessary
    reshaped_data = reshaped_data.fillna(value=np.nan)

    # Save the reshaped data to a new CSV file
    reshaped_data.to_csv(output_csv_path, index=False)

# Example usage
process_and_reshape_csv('results/accumulated_results_2024-05-29_00-51-36.csv', 'reshaped_output.csv')

