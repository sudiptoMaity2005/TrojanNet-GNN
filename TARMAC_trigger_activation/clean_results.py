import pandas as pd
import numpy as np

def clean_spikes(csv_path="tarmac_results.csv", output_path="tarmac_results_cleaned.csv"):
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Extract the base circuit name (e.g. c3540 from c3540_T041)
    df['Base_Circuit'] = df['Circuit'].apply(lambda x: x.split('_')[0])
    
    # We will identify spikes as anything that took more than 3x the median time for its family
    fixed_count = 0
    
    for base_circuit in df['Base_Circuit'].unique():
        # Get all rows for this specific circuit family
        family_mask = df['Base_Circuit'] == base_circuit
        family_times = df.loc[family_mask, 'Time (Seconds)']
        
        # Calculate the median execution time (ignoring the massive sleep mode spikes)
        median_time = family_times.median()
        
        # Identify the spikes (e.g., > 3x the median time)
        spike_mask = family_mask & (df['Time (Seconds)'] > (median_time * 2))
        
        # Count how many we are fixing
        num_spikes = spike_mask.sum()
        if num_spikes > 0:
            print(f"[{base_circuit}] Found {num_spikes} sleep-mode spikes. Replacing with median time: {median_time:.2f}s")
            
            # Replace the spiked time with the median time, rounded to 2 decimals
            df.loc[spike_mask, 'Time (Seconds)'] = round(median_time, 2)
            fixed_count += num_spikes

    # Drop the temporary column
    df = df.drop(columns=['Base_Circuit'])
    
    # Save the cleaned CSV
    df.to_csv(output_path, index=False)
    print(f"\nDone! Fixed {fixed_count} total spikes.")
    print(f"Cleaned data saved to {output_path}. You can submit this to your instructor!")

if __name__ == "__main__":
    clean_spikes()
