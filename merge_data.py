import json
import csv

# Read the JSON file
with open(r'c:\Users\Bende\Documents\Orient-Anything-V2\output\inference_with_ground_truth_final.json', 'r') as f:
    json_data = json.load(f)

# Create a mapping from position_id to ground_truth_heading_deg
ground_truth_map = {}
for prediction in json_data['predictions']:
    position_id = prediction['position_id']
    ground_truth_heading = prediction['ground_truth_heading_deg']
    ground_truth_map[position_id] = ground_truth_heading

# Read the CSV file
input_csv = r'c:\Users\Bende\Documents\Orient-Anything-V2\output\batch_results.csv'
output_csv = r'c:\Users\Bende\Documents\Orient-Anything-V2\output\comparison.csv'

rows = []
with open(input_csv, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Create new CSV with ground_truth_heading_deg added after rotation
with open(output_csv, 'w', newline='') as f:
    # Extract position_id from filename
    fieldnames = ['filename', 'azimuth', 'polar', 'rotation', 'ground_truth_heading_deg', 'num_directions', 'inference_time', 'render_time', 'total_time']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in rows:
        # Extract position_id from filename (e.g., "position_009.png" -> "position_009")
        filename = row['filename']
        position_id = filename.replace('.png', '')
        
        # Get ground_truth_heading_deg from the map
        ground_truth = ground_truth_map.get(position_id, '')
        
        # Create new row with ground_truth_heading_deg after rotation
        new_row = {
            'filename': row['filename'],
            'azimuth': row['azimuth'],
            'polar': row['polar'],
            'rotation': row['rotation'],
            'ground_truth_heading_deg': ground_truth,
            'num_directions': row['num_directions'],
            'inference_time': row['inference_time'],
            'render_time': row['render_time'],
            'total_time': row['total_time']
        }
        writer.writerow(new_row)

print(f"Successfully created {output_csv}")
print(f"Merged {len(rows)} rows with ground truth heading data")
