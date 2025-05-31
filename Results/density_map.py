import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_joint_kde(matrix_path, title, filename):
    # Load matrix
    df = pd.read_csv(matrix_path, index_col=0)
    matrix = df.values

    # Get indices of non-NaN values
    row_idx, col_idx = np.where(~np.isnan(matrix))
    data = pd.DataFrame({'row': row_idx, 'col': col_idx})

    # Plot joint KDE
    g = sns.jointplot(data=data, x="col", y="row", kind="kde", fill=True, cmap='mako')
    g.ax_joint.invert_yaxis()  # To match matrix/heatmap orientation
    g.set_axis_labels("Feature (column)", "Feature (row)", fontsize=14)
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    # Remove axis ticks for a clean look
    # g.ax_joint.set_xticks([])
    # g.ax_joint.set_yticks([])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# Example usage:
plot_joint_kde('feature_density_matrix_cancer.csv', '2D KDE of Data Locations (Cancer)', 'cancer_density_joint_kde.png')
plot_joint_kde('feature_density_matrix_stroke.csv', '2D KDE of Data Locations (Stroke)', '_density_joint_kde.png')
