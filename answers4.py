import os
import pandas as pd
from tqdm import tqdm
import re

def extract_malicious_days(data_dir, output_file='answers4.csv'):
    """
    Extracts malicious activity dates from multiple files with mixed date formats.
    Assumes:
    - No header
    - Column 3 (index 2) = timestamp
    - User ID and Scenario number are extracted from filename, e.g. r4.2-1-BLS0678.csv -> scenario=1, user=BLS0678
    """
    all_rows = []

    # Regex: captures scenario number and user ID
    # Example: r4.2-1-BLS0678.csv -> scenario=1, user=BLS0678
    filename_pattern = re.compile(r'r4\.2-(\d+)-([A-Z0-9]+)\.csv$', re.IGNORECASE)

    for fname in tqdm(os.listdir(data_dir)):
        if not fname.endswith('.csv'):
            continue

        match = filename_pattern.search(fname)
        if not match:
            print(f"Warning: could not parse scenario/user from filename {fname}")
            continue

        scenario = match.group(1)
        user = match.group(2)

        file_path = os.path.join(data_dir, fname)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue  # skip malformed rows

                raw_date = parts[2].strip()
                parsed = pd.to_datetime(raw_date, errors='coerce')

                if pd.notnull(parsed):
                    all_rows.append((scenario, user, parsed.date()))

    # Build DataFrame
    df = pd.DataFrame(all_rows, columns=['scenario', 'user', 'date']).drop_duplicates()
    df.to_csv(output_file, index=False)
    print(f"Created {output_file} with {len(df)} rows.")

if __name__ == '__main__':
    data_dir = '../../data/r4.2/r.4.2answers'  # <-- Update this
    extract_malicious_days(data_dir)
