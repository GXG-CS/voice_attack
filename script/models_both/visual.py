import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('model_evaluation_results.csv')

# Extract unique categories and models
categories = df['Dataset Category'].unique()
models = df['Model'].unique()

# Initialize a dictionary to hold the data
data = {model: [] for model in models}

# Populate the dictionary with accuracy values for each model within each category
for category in categories:
    filtered_df = df[df['Dataset Category'] == category]
    for model in models:
        model_df = filtered_df[filtered_df['Model'] == model]
        if not model_df.empty:
            data[model].append(model_df['Accuracy'].values[0])
        else:
            data[model].append(0)  # If the model-category combination doesn't exist

# Define the position of the bars
bar_width = 0.15  # Adjust the width as necessary
index = np.arange(len(categories))

# Plotting the bars
fig, ax = plt.subplots(figsize=(15, 8))

for i, (model, accuracies) in enumerate(data.items()):
    ax.bar(index + i * bar_width, accuracies, bar_width, label=model)

# Adding labels and title
ax.set_xlabel('Dataset Category')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(categories, rotation=45)
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of ML Models Across Different Dataset Categories')

# Adding legend
ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()
