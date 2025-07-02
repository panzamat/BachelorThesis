""" Compute target-wise metrics usin the iou thresholding method 
Filter out aneurysms < 1mm 
Compute summary table with 95% confidence intervals"""


from Trying_AUC_ROC import compute_validation
from compute_all_center_of_mass import aggregate_metrics
import os
import pandas as pd
import numpy as np
np.bool = np.bool_
from tqdm import tqdm

"""
def process_files(pred_files, gt_files, component_csv_file_name):
     Process multiple NIfTI file pairs to calculate and aggregate metrics.
    all_results = []
    for pred_file, gt_file in tqdm(zip(pred_files, gt_files), total=len(pred_files), desc="Processing Files"):
        file_results = compute_validation(pred_file, gt_file)
        for result in file_results:
            all_results.append(result)  # Using append instead of extend
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    #df.to_csv(component_csv_file_name, index=False)
    #print(f"Component-wise results saved to {component_csv_file_name}")

    return results_df

def process_files(pred_files, gt_files):
    Process multiple NIfTI file pairs to calculate and aggregate metrics.
    all_results = []
    for pred_file, gt_file in tqdm(zip(pred_files, gt_files), total=len(pred_files), desc="Processing Files"):
        file_results = compute_validation(pred_file, gt_file)

        # Add DSC to each component-wise result if it's a TP
        for result in file_results:
            if result['Match Type'] == 'TP':
                intersection = result.get('Intersection')
                pred_size = result.get('Pred Size')
                gt_size = result.get('GT Size')
                if intersection is not None and pred_size and gt_size:
                    denom = pred_size + gt_size
                    result['DSC'] = 2 * intersection / denom if denom > 0 else 0
                else:
                    result['DSC'] = np.nan
            else:
                result['DSC'] = np.nan

        all_results.extend(file_results)

    # Convert results to DataFrame
    df = pd.DataFrame(all_results)

    return df
"""
def process_files(pred_files, gt_files):
    """ Process multiple NIfTI file pairs to calculate and aggregate metrics. """
    all_results = []
    for pred_file, gt_file in tqdm(zip(pred_files, gt_files), total=len(pred_files), desc="Processing Files"):
        filename = os.path.basename(pred_file)
        file_results = compute_validation(pred_file, gt_file)

        for result in file_results:
            result["file_name"] = filename  # 🔧 wichtig für aggregation

            # Add DSC to each component-wise result if it's a TP
            if result['Match Type'] == 'TP':
                intersection = result.get('Intersection')
                pred_size = result.get('Pred Size')
                gt_size = result.get('GT Size')
                if intersection is not None and pred_size and gt_size:
                    denom = pred_size + gt_size
                    result['DSC'] = 2 * intersection / denom if denom > 0 else 0
                else:
                    result['DSC'] = np.nan
            else:
                result['DSC'] = np.nan

            all_results.append(result)

    # Convert results to DataFrame
    df = pd.DataFrame(all_results)
    return df


def summarize_metrics(all_metrics_df, dataset):
    """
    Calculates the mean and standard deviation for each metric in a DataFrame,
    formats the summary as "mean (mean - sd, mean + sd)", and saves the result to a CSV file.

    Parameters:
    - all_metrics_df: A pandas DataFrame containing the metrics to be summarized.

    Returns:
    - None
    """
    # Calculate mean and standard deviation for each metric
    metrics_summary = all_metrics_df.agg(['mean', 'std']).transpose()

    # Format the summary
    metrics_summary['formatted_result'] = metrics_summary.apply(
        lambda row: f"{row['mean']:.2f} ({row['mean'] - row['std']:.2f} - {row['mean'] + row['std']:.2f})", axis=1)

    # Save the formatted summary to a CSV file
    metrics_summary['formatted_result'].to_csv(f"formatted_summary_{dataset}_iou_after_filtering.csv", header=True)


def summarize_metrics_CI(all_metrics_df, dataset):
    metrics_summary = all_metrics_df.agg(['mean', 'std', 'count']).transpose()

    metrics_summary['lower_bound'] = metrics_summary.apply(
        lambda row: row['mean'] - (1.96 * (row['std'] / (row['count'] ** 0.5))), axis=1)
    metrics_summary['upper_bound'] = metrics_summary.apply(
        lambda row: row['mean'] + (1.96 * (row['std'] / (row['count'] ** 0.5))), axis=1)

    metrics_summary['formatted_result'] = metrics_summary.apply(
        lambda row: f"{row['mean']:.2f} ({row['lower_bound']:.2f} - {row['upper_bound']:.2f})", axis=1)

    metrics_summary['formatted_result'].to_csv(f"formatted_summary_{dataset}_95CI_iou.csv", header=True)
    print(f"Formatted summary saved to formatted_summary_{dataset}_95CI_iou.csv")

if __name__ == "__main__":
    dataset = "CT"
    folds = range(5)
    all_metrics_CT = []  # List to store aggregated metrics for all folds

    with tqdm(total=len(folds), desc=f"Processing {dataset}") as pbar:
        for i in folds:
            pred_files_path = f'/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/nnUNet_data/nnUNet_results/Dataset059_IA/postprocessed/postprocessed_f{i}'
            gt_files_path = f'/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/nnUNet_data/nnUNet_raw/Dataset059_IA/labelsTs_internal'

            # 🔐 Sicheres Matching per Dateinamen
            gt_dict = {
                f: os.path.join(gt_files_path, f)
                for f in os.listdir(gt_files_path)
                if f.endswith('.nii.gz')
            }
            pred_filenames = [
                f for f in os.listdir(pred_files_path)
                if f.endswith('.nii.gz') and f in gt_dict
            ]

            pred_files = [os.path.join(pred_files_path, f) for f in pred_filenames]
            gt_files = [gt_dict[f] for f in pred_filenames]

            # Optional: Warnung für nicht gematchte Dateien
            missing = [f for f in os.listdir(pred_files_path) if f.endswith('.nii.gz') and f not in gt_dict]
            if missing:
                print(f"⚠️ Fold {i}: {len(missing)} prediction files had no matching GT and were skipped:")
                for f in missing:
                    print(f"   - {f}")

            # Output-Dateien
            aggregated_csv_file_name = f"aggregated_{dataset}_f{i}_iou_after_filtering_CT.csv"
            component_csv_file_name = f"component_{dataset}_f{i}_iou_after_filtering_CT.csv"

            # Verarbeitung
            results_df = process_files(pred_files, gt_files)
            aggregated_metrics = aggregate_metrics(results_df, aggregated_csv_file_name)

            print(f"✅ Processed {dataset} fold {i}")
            aggregated_metrics_df = pd.DataFrame([aggregated_metrics])
            all_metrics_CT.append(aggregated_metrics_df)
            pbar.update()

    all_metrics_df = pd.concat(all_metrics_CT, ignore_index=True)
    all_metrics_df.to_csv(f"all_aggregated_{dataset}_iou_after_filtering_CT.csv", index=False)
    summarize_metrics_CI(all_metrics_df, dataset)

"""
if __name__ == "__main__":
    dataset = "MR"
    folds = range(5)
    all_metrics_MR = []  # List to store aggregated metrics for all folds

    with tqdm(total=len(folds), desc=f"Processing {dataset}") as pbar:
        for i in folds:
            pred_files_path = f'/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/nnUNet_data/nnUNet_results/Dataset060_IA/postprocessed/postprocessed_f{i}'
            gt_files_path = f'/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/nnUNet_data/nnUNet_raw/Dataset060_IA/labelsTs'

            # 🔐 Sicheres Matching per Dateinamen
            gt_dict = {
                f: os.path.join(gt_files_path, f)
                for f in os.listdir(gt_files_path)
                if f.endswith('.nii.gz')
            }
            pred_filenames = [
                f for f in os.listdir(pred_files_path)
                if f.endswith('.nii.gz') and f in gt_dict
            ]

            pred_files = [os.path.join(pred_files_path, f) for f in pred_filenames]
            gt_files = [gt_dict[f] for f in pred_filenames]

            # Optional: Warnung für nicht gematchte Dateien
            missing = [f for f in os.listdir(pred_files_path) if f.endswith('.nii.gz') and f not in gt_dict]
            if missing:
                print(f"⚠️ Fold {i}: {len(missing)} prediction files had no matching GT and were skipped:")
                for f in missing:
                    print(f"   - {f}")

            # Output-Dateien
            aggregated_csv_file_name = f"aggregated_{dataset}_f{i}_iou_after_filtering.csv"
            component_csv_file_name = f"component_{dataset}_f{i}_iou_after_filtering.csv"

            # Verarbeitung
            results_df = process_files(pred_files, gt_files)
            aggregated_metrics = aggregate_metrics(results_df, aggregated_csv_file_name)

            print(f"✅ Processed {dataset} fold {i}")
            aggregated_metrics_df = pd.DataFrame([aggregated_metrics])
            all_metrics_MR.append(aggregated_metrics_df)
            pbar.update()

    all_metrics_df = pd.concat(all_metrics_MR, ignore_index=True)
    all_metrics_df.to_csv(f"all_aggregated_{dataset}_iou_after_filtering.csv", index=False)
    summarize_metrics_CI(all_metrics_df, dataset)


if __name__ == "__main__":
    dataset = "MR"
    folds = range(5)
    all_metrics_MR = []  # List to store aggregated metrics for all folds
    with tqdm(total=len(folds), desc=f"Processing {dataset}") as pbar:
        for i in folds:
            pred_files_path = f'/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/nnUNet_data/nnUNet_results/Dataset060_IA/postprocessed/postprocessed_f{i}'
            gt_files_path = f'/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/nnUNet_data/nnUNet_raw/Dataset060_IA/labelsTs'
            gt_files = [os.path.join(gt_files_path, f) for f in os.listdir(gt_files_path) if f.endswith('.nii.gz')]
            pred_files = [os.path.join(pred_files_path, f) for f in os.listdir(pred_files_path) if f.endswith('.nii.gz')]
            aggregated_csv_file_name = f"aggregated_{dataset}_f{i}_iou_after_filtering.csv"
            component_csv_file_name = f"component_{dataset}_f{i}_iou_after_filtering.csv"
            results_df = process_files(pred_files, gt_files)
            aggregated_metrics = aggregate_metrics(results_df,aggregated_csv_file_name)
            print(f"Processed {dataset} fold {i}")
            pbar.update()
            aggregated_metrics_df = pd.DataFrame([aggregated_metrics])  # Convert dict to DataFrame
            all_metrics_MR.append(aggregated_metrics_df)  # Append DataFrame instead of dict
    
    # Assuming all_metrics is a list of DataFrames
    all_metrics_df = pd.concat(all_metrics_MR, ignore_index=True)
    all_metrics_df.to_csv(f"all_aggregated_{dataset}_iou_after_filtering.csv", index=False)
    #summarize_metrics(all_metrics_df, dataset)
    summarize_metrics_CI(all_metrics_df, dataset)
        
    dataset = "CT"
    all_metrics_CT = []  # List to store aggregated metrics for all folds
    folds = range(5)
    with tqdm(total=len(folds), desc=f"Processing {dataset}") as pbar:
        for i in folds:
            pred_files_path = f'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset059_IA/postprocessed/postprocessed_f{i}'
            gt_files_path = f'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset059_IA/labelsTs_internal'
            gt_files = [os.path.join(gt_files_path, f) for f in os.listdir(gt_files_path) if f.endswith('.nii.gz')]
            pred_files = [os.path.join(pred_files_path, f) for f in os.listdir(pred_files_path) if f.endswith('.nii.gz')]
            aggregated_csv_file_name = f"aggregated_{dataset}_f{i}_iou_after_filtering.csv"
            component_csv_file_name = f"component_{dataset}_f{i}_iou_after_filtering.csv"
            results_df = process_files(pred_files, gt_files)
            aggregated_metrics = aggregate_metrics(results_df,aggregated_csv_file_name)
            print(f"Processed {dataset} fold {i}")
            pbar.update()
            aggregated_metrics_df = pd.DataFrame([aggregated_metrics])  # Convert dict to DataFrame
            all_metrics_CT.append(aggregated_metrics_df)  # Append DataFrame instead of dict

     # Assuming all_metrics is a list of DataFrames
    all_metrics_df = pd.concat(all_metrics_CT, ignore_index=True)
    all_metrics_df.to_csv(f"all_aggregated_{dataset}_iou_after_filtering.csv", index=False)
    #summarize_metrics(all_metrics_df, dataset)
    summarize_metrics_CI(all_metrics_df, dataset)
"""    

