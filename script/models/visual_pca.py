# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the results from the CSV file
# results_df = pd.read_csv('model_evaluation_results_pca.csv')

# # Setting up the visualization style
# sns.set(style="whitegrid")

# # Plotting Top-5 Accuracy for different PCA Components
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=results_df, x='PCA Components', y='Top-5 Accuracy', hue='Model', marker='o')
# plt.title('Top-5 Accuracy for Different PCA Components')
# plt.xlabel('Number of PCA Components')
# plt.ylabel('Top-5 Accuracy')
# plt.legend(title='Classifier')
# plt.show()

# # Plotting Accuracy for different PCA Components
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=results_df, x='PCA Components', y='Accuracy', hue='Model', marker='o')
# plt.title('Accuracy for Different PCA Components')
# plt.xlabel('Number of PCA Components')
# plt.ylabel('Accuracy')
# plt.legend(title='Classifier')
# plt.show()

# # Plotting MCC for different PCA Components
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=results_df, x='PCA Components', y='MCC', hue='Model', marker='o')
# plt.title('Matthews Correlation Coefficient (MCC) for Different PCA Components')
# plt.xlabel('Number of PCA Components')
# plt.ylabel('MCC')
# plt.legend(title='Classifier')
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results from the CSV file
results_df = pd.read_csv('model_evaluation_results_pca.csv')

# Setting up the visualization style
sns.set(style="whitegrid")

# Create a figure and a set of subplots
plt.figure(figsize=(14, 8))

# Iterate over each model type to plot each one on the same axes
for model in results_df['Model'].unique():
    model_data = results_df[results_df['Model'] == model]
    
    # Plotting each metric with different line styles and markers
    # sns.lineplot(x=model_data['PCA Components'], y=model_data['Top-5 Accuracy'], marker='o', linestyle='-', label=f'Top-5 Acc - {model}')
    sns.lineplot(x=model_data['PCA Components'], y=model_data['Accuracy'], marker='s', linestyle='--', label=f'Acc - {model}')
    # sns.lineplot(x=model_data['PCA Components'], y=model_data['MCC'], marker='^', linestyle=':', label=f'MCC - {model}')

plt.title('Performance Metrics for Different PCA Components Across Models')
plt.xlabel('Number of PCA Components')
plt.ylabel('Metric Value')
plt.legend(title='Metrics and Models', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.show()
