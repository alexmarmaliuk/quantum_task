import csv

input_file = 'data/gpt_dataset.csv'
output_file = 'data/gpt_datasett.csv'

with open(input_file, 'r', newline='', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        # Skip empty rows
        if not row:
            writer.writerow('')
            continue
        
        # Trim trailing spaces
        trimmed_row = [cell.strip() for cell in row]
        
        # Check if the first cell contains 'MOUNTAIN' and replace with 'mountain'
        if 'MOUNTAIN' in trimmed_row[0][:8]:
            trimmed_row[0] = trimmed_row[0][:8].replace('MOUNTAIN', 'mountain')
        
        writer.writerow(trimmed_row)

print(f"Trailing spaces trimmed and 'MOUNTAIN' replaced with 'mountain' in the first cell, saved to {output_file}.")
