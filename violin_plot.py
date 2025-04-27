import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Set the backend to Agg to avoid display issues
plt.switch_backend('Agg')

# Set style and font
plt.style.use('seaborn')
plt.rcParams['font.family'] = 'Arial'

# Select numerical features for visualization
numerical_features = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']

# Load the data
df = pd.read_csv('kidney_disease.csv')

# Clean the data - replace empty strings with NaN and drop rows with NaN
df = df.replace('', pd.NA)
df = df.dropna(subset=numerical_features)

# Create a figure with multiple subplots
plt.figure(figsize=(15, 10))

# Create violin plots for each numerical feature
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 4, i)
    sns.violinplot(x='classification', y=feature, data=df)
    plt.title(f'Distribution of {feature}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('kidney_disease_violin_plots.png', dpi=300, bbox_inches='tight')
print("Violin plots have been saved to 'kidney_disease_violin_plots.png'") 