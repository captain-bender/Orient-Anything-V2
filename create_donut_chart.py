import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('output/comparison.csv')

# Categorize the axis_direction_error values
categories = {
    'Excellent (0-3°)': 0,
    'Very Good (3-5°)': 0,
    'Good (5-10°)': 0,
    'Not Reliable (>10°)': 0,
    'No Inference (NaN)': 0
}

for error in df['axis_direction_error']:
    if pd.isna(error):
        categories['No Inference (NaN)'] += 1
    elif error <= 3:
        categories['Excellent (0-3°)'] += 1
    elif error <= 5:
        categories['Very Good (3-5°)'] += 1
    elif error <= 10:
        categories['Good (5-10°)'] += 1
    else:
        categories['Not Reliable (>10°)'] += 1

# Prepare data for donut chart
labels = list(categories.keys())
sizes = list(categories.values())
colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6']

# Create the figure
fig, ax = plt.subplots(figsize=(10, 8))

# Create donut chart
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 11, 'weight': 'bold'}
)

# Draw circle for donut
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax.add_artist(centre_circle)

# Add title
plt.title('Orientation Prediction Accuracy Distribution', fontsize=14, weight='bold', pad=20)

# Add legend with counts
legend_labels = [f'{label}: {count}' for label, count in zip(labels, sizes)]
plt.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)

plt.tight_layout()

# Save the figure
plt.savefig('output/accuracy_donut_chart.png', dpi=300, bbox_inches='tight')
print("Donut chart saved as 'output/accuracy_donut_chart.png'")

# Print summary statistics
print("\n=== Accuracy Distribution Summary ===")
for label, count in categories.items():
    percentage = (count / len(df)) * 100
    print(f"{label}: {count} ({percentage:.1f}%)")
print(f"\nTotal samples: {len(df)}")
