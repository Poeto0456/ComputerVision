import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
import json

def display_params(params_json_path):
    """
    Read and print training parameters from a params.json file.

    Args:
        params_json_path (str | Path): Path to params.json.
    """
    try:
        with open(params_json_path, 'r') as f:
            params = json.load(f)
        
        print("--- Training Parameters ---")
        for key, value in params.items():
            print(f"{key}: {value}")
        print("-" * 30)

    except FileNotFoundError:
        print(f"Error: params.json not found at '{params_json_path}'")
    except json.JSONDecodeError:
        print(f"Error: params.json at '{params_json_path}' is not valid JSON.")


def display_images(image_paths, titles=None, cols=2, figure_size=(15, 8)):
    """
    Display a list of images.

    Args:
        image_paths (list[str | Path]): List of image paths.
        titles (list[str], optional): Titles for each image.
        cols (int): Number of columns in the grid.
        figure_size (tuple): Matplotlib figure size.
    """
    if not image_paths:
        print("No image paths provided.")
        return
        
    n_images = len(image_paths)
    rows = (n_images + cols - 1) // cols
    fig = plt.figure(figsize=figure_size)

    for i, image_path in enumerate(image_paths):
        try:
            img = Image.open(image_path)
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(img)
            ax.axis('off')
            if titles and i < len(titles):
                ax.set_title(titles[i])
        except FileNotFoundError:
            print(f"Error: File not found at '{image_path}'")
            
    plt.tight_layout()
    plt.show()

def plot_training_results(results_csv_path, save_plot=False):
    """
    Read results CSV and plot training metrics.

    Expects columns like:
      - epoch
      - metrics/precision(B)
      - metrics/recall(B)
      - metrics/mAP50(B)
      - metrics/mAP50-95(B)
      - train/box_loss, train/cls_loss, train/dfl_loss
      - val/box_loss, val/cls_loss, val/dfl_loss

    Args:
        results_csv_path (str | Path): Path to the training results CSV.
        save_plot (bool): If True, save the combined plot as 'custom_plots.png'
                          in the same folder as the CSV.
    """
    try:
        df = pd.read_csv(results_csv_path)
        df.columns = df.columns.str.strip()
        fig, axes = plt.subplots(1, 2, figsize=(20, 7))
        fig.suptitle('Training Overview', fontsize=16)

        # Metrics (Precision, Recall, mAP)
        axes[0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
        axes[0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
        axes[0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
        axes[0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5-0.95')
        axes[0].set_title('Validation Metrics Chart')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True)

        # Loss curves
        axes[1].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
        axes[1].plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss')
        axes[1].plot(df['epoch'], df['train/dfl_loss'], label='Train DFL Loss')
        axes[1].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', linestyle='--')
        axes[1].plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss', linestyle='--')
        axes[1].plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss', linestyle='--')
        axes[1].set_title('Loss Chart (Training & Validation)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_plot:
            save_path = Path(results_csv_path).parent / 'custom_plots.png'
            plt.savefig(save_path)
            print(f"Plot saved to: {save_path}")
            
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: File not found at '{results_csv_path}'")
    except KeyError as e:
        print(f"Error: Missing column {e} in CSV. Please verify column names.")
    
import matplotlib.gridspec as gridspec

def visualize_test_metrics(csv_path):
    """
    Read test_metrics CSV and create a composite figure:
      - Top row: horizontal summary text showing overall metrics ('all')
      - Bottom: two per-class bar charts (mAP@.50-.95 and mAP@.50)

    Args:
        csv_path (str | Path): Path to the test metrics CSV.
    """
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        all_metrics = df[df['class'] == 'all']
        per_class_metrics = df[df['class'] != 'all'].copy()

        # Layout: top row for text, bottom row for two charts
        fig = plt.figure(figsize=(22, 18))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 9])
        
        fig.suptitle('Test Set Evaluation', fontsize=24, y=0.98)

        # Top: summary 
        ax_text = fig.add_subplot(gs[0, :])
        ax_text.axis('off') 

        if not all_metrics.empty:
            metrics_values = all_metrics.iloc[0]
            summary_text = (
                f"Overall Metrics:    "
                f"Precision: {metrics_values['precision']:.4f}   |   "
                f"Recall: {metrics_values['recall']:.4f}   |   "
                f"mAP@50: {metrics_values['mAP50']:.4f}   |   "
                f"mAP@50-95: {metrics_values['mAP50-95']:.4f}"
            )
            ax_text.text(0.5, 0.5, summary_text, 
                         ha='center', va='center', fontsize=18, 
                         fontfamily='monospace')
        else:
            ax_text.text(0.5, 0.5, "Not find result 'all'",
                         ha='center', va='center', fontsize=16)

        # Bottom: per-class charts
        if not per_class_metrics.empty:
            per_class_metrics.sort_values('mAP50-95', ascending=False, inplace=True)

            # Left: mAP50-95
            ax1 = fig.add_subplot(gs[1, 0])
            sns.barplot(x='mAP50-95', y='class', data=per_class_metrics, ax=ax1, palette='plasma')
            ax1.set_title('mAP @ .50-.95', fontsize=16)
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Class')
            ax1.grid(axis='x', linestyle='--', alpha=0.7)
            ax1.set_xlim(right=per_class_metrics['mAP50-95'].max() * 1.1) 
            for p in ax1.patches:
                ax1.text(p.get_width() * 1.01, p.get_y() + p.get_height() / 2,
                         f'{p.get_width():.2f}', va='center')

            # Right: mAP50
            ax2 = fig.add_subplot(gs[1, 1])
            sns.barplot(x='mAP50', y='class', data=per_class_metrics, ax=ax2, palette='cividis')
            ax2.set_title('mAP @ .50', fontsize=16)
            ax2.set_xlabel('Value')
            ax2.set_ylabel('') 
            ax2.set_yticklabels([]) 
            ax2.grid(axis='x', linestyle='--', alpha=0.7)
            ax2.set_xlim(right=per_class_metrics['mAP50'].max() * 1.1)
            for p in ax2.patches:
                ax2.text(p.get_width() * 1.01, p.get_y() + p.get_height() / 2,
                         f'{p.get_width():.2f}', va='center')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    except FileNotFoundError:
        print(f"Error: File not found at '{csv_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")