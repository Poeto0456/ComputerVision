import cv2
from ultralytics import YOLO
from pathlib import Path
import cv2 
import pandas as pd

# LOAD & INFERENCE HELPERS

def load_yolo_model(weights_path):
    """
    Load a YOLOv8 model from a given weights file
    Args: weights_path (str | Path): Path to the .pt weights file
    Returns: YOLO | None: Model object if loaded successfully, otherwise None
    """
    weights_file = Path(weights_path)
    if not weights_file.exists():
        print(f"Error: Weights file not found at '{weights_path}'")
        return None
        
    try:
        model = YOLO(weights_file)
        print(f"Model loaded successfully from:{weights_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_on_source(model, source_path, conf_threshold=0.5):
    """
    Run inference on an image, video, or webcam.

    Args:
        model (YOLO): Loaded YOLOv8 model.
        source_path (str | int): Image/video path or 0 for webcam.
        conf_threshold (float): Confidence threshold.

    Returns: generator | None: Prediction results generator, or None if failed.
    """
    if model is None:
        print("Error: Invalid model. Cannot run inference.")
        return None
        
    try:
        results = model.predict(source=source_path, conf=conf_threshold, stream=True)
        return results
    except Exception as e:
        print(f"Lỗi trong quá trình inference: {e}")
        return None

def display_results(results):
    """
    Display YOLOv8 inference results with OpenCV.
    Args: results (generator): Output from predict_on_source().
    """
    if results is None:
        print("No results to display.")
        return
        
    print("Displaying results... Press 'q' to quit.")
    try:
        for r in results:
            im_array = r.plot()  
            cv2.imshow("YOLOv8 Inference", im_array)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        print("Display window closed.")

# SAVE / EXPORT HELPERS

def save_results_from_folder(results, output_dir):
    """
    Save inference results (images with boxes) to an output folder.

    Args:
        results (generator): Output from model.predict(stream=True).
        output_dir (str | Path): Target directory to save results.
    """
    if results is None:
        print("No results to save.")
        return
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {output_path} ...")
    
    count = 0
    for r in results:
        original_filename = Path(r.path).name
        save_path = output_path / original_filename
        im_array = r.plot()
        cv2.imwrite(str(save_path), im_array)
        count += 1

    print(f"Done! Saved {count} images to {output_path}")

# EVALUATION & METRICS EXPORT

def run_evaluation(weights_path, data_yaml_path, output_csv_path, exp_dir_path):
    """
    Evaluate model performance on test split and export metrics to CSV.

    Args:
        weights_path (str | Path): Path to weights (.pt file).
        data_yaml_path (str | Path): Path to data.yaml.
        output_csv_path (str | Path): Output path for the CSV file.
        exp_dir_path (Path): Experiment directory (contains weights).

    Returns: metrics | None: YOLOv8 metrics object if successful, else None.
    """
    
    # Kiểm tra các file đầu vào
    if not Path(weights_path).exists():
        print(f"Error: Weights not found at '{weights_path}'")
        return None
    if not Path(data_yaml_path).exists():
        print(f"Error: Data YAML not found at  '{data_yaml_path}'")
        return None

    try:
        model = YOLO(weights_path)
        
        print("\nStarting evaluation on test split...")
        
        TEST_RUN_NAME = f"{exp_dir_path.name}" 
        metrics = model.val(
            data=str(data_yaml_path), 
            split='test',
            project=str(exp_dir_path.parent),
            name=TEST_RUN_NAME,
            exist_ok=True,
            save_json=True
        )
        
        test_plots_dir = exp_dir_path.parent / TEST_RUN_NAME
        print(f"Evaluation plots saved to: {test_plots_dir}")
        print(f"\nEvaluation complete! mAP50-95: {metrics.box.map:.4f}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None

    # --- Extract data and save CSV ---
    class_names = metrics.names
    results_data = []
    
    # Extract metrics for each class
    for i, class_name in class_names.items():
        results_data.append({
            'class': class_name,
            'precision': metrics.class_result(i)[0],
            'recall': metrics.class_result(i)[1],
            'mAP50': metrics.class_result(i)[2],
            'mAP50-95': metrics.class_result(i)[3]
        })
        
    # Add summary row
    results_data.append({
        'class': 'all',
        'precision': metrics.box.mp,
        'recall': metrics.box.mr,
        'mAP50': metrics.box.map50,
        'mAP50-95': metrics.box.map
    })

    # --- Create DataFrame and save to CSV ---
    try:
        df = pd.DataFrame(results_data)
        for col in ['precision', 'recall', 'mAP50', 'mAP50-95']:
            df[col] = pd.to_numeric(df[col], errors='coerce').apply(lambda x: f"{x:.4f}") 
            
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        
        print(f"\n[OK] Test results saved to: {output_csv_path}")
        print("\nDetailed metrics:")
        print(df.to_string(index=False))
        
    except Exception as e:
        print(f"Error saving CSV file: {e}")
        
    return metrics 