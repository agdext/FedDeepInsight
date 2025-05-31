import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
cancer = pd.read_csv('trustworthiness_scores_cancer.csv')
stroke = pd.read_csv('trustworthiness_scores_stroke.csv')

# Set seaborn style for publication-quality plots
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.2)

def plot_trustworthiness(data, title, filename):
    # Melt the dataframe for seaborn compatibility
    df_melted = data.melt(id_vars='k', value_vars=['t-SNE Score', 'PCA Score'],
                          var_name='Method', value_name='Trustworthiness Score')
    # Rename methods for cleaner legend
    df_melted['Method'] = df_melted['Method'].replace({'t-SNE Score': 't-SNE', 'PCA Score': 'PCA'})
    
    plt.figure(figsize=(8, 6))
    # Lineplot with markers for each method
    sns.lineplot(
        data=df_melted, x='k', y='Trustworthiness Score', hue='Method',
        marker='o', linewidth=4, markersize=12
    )
    # plt.title(title, fontsize=24, pad=15)
    plt.xlabel('k(Number of Neighbors)', fontsize=22)
    plt.ylabel('Trustworthiness Score', fontsize=14)
    plt.legend(title='Method', fontsize=18, title_fontsize=18, loc='center right', bbox_to_anchor=(1,0.35))
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# Plot cancer dataset
plot_trustworthiness(
    cancer,
    'Trustworthiness Scores for t-SNE and PCA (Cancer)',
    'trustworthiness_comparison_cancer_seaborn.png'
)

# Plot stroke dataset
plot_trustworthiness(
    stroke,
    'Trustworthiness Scores for t-SNE and PCA (Stroke)',
    'trustworthiness_comparison_stroke_seaborn.png'
)
