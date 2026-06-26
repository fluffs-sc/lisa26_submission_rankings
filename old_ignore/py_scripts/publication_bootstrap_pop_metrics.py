#!/usr/bin/env python3
"""
Model Submission Bootstrap Reliability Metrics Pipeline for Publication

Input: folder containing separate CSVs for each task, where each CSV is a cleaned long-form CSV of raw performance metric values for a single task
Function performs: BCa bootstrap of Hodges-Lehmann median estimator per model per metric, computing standard error, empirical bias, and 99% BCa confidence intervals
Output: bootstrap reliability metrics for each model per task in CSV
"""

#########################################################################
##### import libraries ##################################################
#########################################################################

import json
import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from itertools import combinations_with_replacement

#########################################################################
##### global variables ##################################################
#########################################################################

JSON_PATH = "./metric_rank_directions.json"
OUTPUT_BOOTSTRAP_DIR = "./bootstrap_reliability_reports"
RAW_METRICS_FOLDER_PATH = "./task_master_metrics"
task_csv_path_list = glob.glob(f"{RAW_METRICS_FOLDER_PATH}/*.csv")
BOOTSTRAP_RESAMPLES = 1000
CONFIDENCE_LEVEL = 0.99  # 1-alpha level for BCa confidence interval calculation

#########################################################################
##### func: load json metric rank direction config ######################
#########################################################################
def load_metric_directions(metric_directions_path):
    """
    load json file of metric rank directions (for which metric is higher or lower considered a 'better' value).
    """
    if not os.path.exists(metric_directions_path):
        print(f"Critical error: metric rank directions JSON missing at {metric_directions_path}")
        sys.exit(1)
    with open(metric_directions_path, 'r') as f:
        return json.load(f)

#########################################################################
##### func: hodges-lehmann median estimator #############################
#########################################################################
def hodges_lehmann(x):
    """
    Func to compute Hodges-Lehmann median estimator
    - Using HL median estimator as Bootstrap statistic, alternative to sample mean/median to accomodate for the small sample size and non-gaussian distribution
    - HL median estimator = median of all pairwise averages
    - axis=-1 required for scipy bootstrap vectorized interface.
    """
    # iterate pairwise averages across last axis for scipy bootstrap compatibility
    pairs = np.array([
        (x[..., i] + x[..., j]) / 2.0
        for i, j in combinations_with_replacement(range(x.shape[-1]), 2)
    ])
    return np.median(pairs, axis=0)

#########################################################################
##### func: compute BS metrics for single model+metric data vector in given task ##########
#########################################################################
def compute_bootstrap_metrics(data_vector):
    """
    Compute Bias-Corrected and Accelerated bootstrap metrics using Hodges-Lehmann median estimator.
        Returns 
        - standard error, 
        - empirical bias, and 
        - 99% BCa confidence interval.
    On BCa failure, returns NaNs with a printed warning rather than zeros or program exit.
    """
    data = np.array(data_vector, dtype=float)

    # compute HL estimate on original sample as point estimate and bias reference
    hl_original = hodges_lehmann(data.reshape(1, -1)).item()

    try:
        res = bootstrap(
            (data,),
            hodges_lehmann,
            vectorized=True,
            n_resamples=BOOTSTRAP_RESAMPLES,
            confidence_level=CONFIDENCE_LEVEL,
            method='bca'
        )

        # extract bootstrap metrics
        standard_error = res.standard_error
        ci_lower = res.confidence_interval.low
        ci_upper = res.confidence_interval.high
            # calc empirical bias= (mean of BS HL distribution) - (og sample HL estimate)
        empirical_bias = np.mean(res.bootstrap_distribution) - hl_original

    except ValueError as e:
        # if BCa fail b/c of small samples (e.g. all identical values, zero jackknife variance), NaN is reported explicitly instead of zeros to avoid misleading clinical readers
        print(f"Warning: BCa bootstrap failed ({e}). Returning NaNs for this metric/model combination.")
        standard_error = np.nan
        ci_lower = np.nan
        ci_upper = np.nan
        empirical_bias = np.nan

    return {
        "HL_Median_Estimate": hl_original,
        "Standard_Error": standard_error,
        "Empirical_Bias": empirical_bias,
        f"{int(CONFIDENCE_LEVEL * 100)}%_BCa_CI_Lower": ci_lower,
        f"{int(CONFIDENCE_LEVEL * 100)}%_BCa_CI_Upper": ci_upper
    }


#########################################################################
##### func: BS metrics for given single task #############
#########################################################################
def task_bootstrap_reliability(df_task, task_id):
    """
    For each task:
    - does task use structure+metric names (METRICTYPE_Structure, Task 2) or atomic metric names (Tasks 1a & 1b)
    - (tasks 1a, 1b, 2) For each model and each metric, compute BCa BS metrics using HL estimator as stat (use compute_bootstrap_metrics func defined above)
    - (only task 2), additionally compute OVERALL_COMBINED BS metrics by averaging across structures per subject before bootstrapping
    """

    bootstrap_rows = []

    unique_models = df_task['Model_ID'].unique()

    for model_id in unique_models:
        df_model = df_task[df_task['Model_ID'] == model_id]
        unique_metrics = df_model['Performance_Metric'].unique()

        # detect metric naming convention (METRICTYPE_Structure)
        is_compound_task = any('_' in m for m in unique_metrics)

        # task 2 detected with structure+metric names
        if is_compound_task:
            # parse base metric types and anatomical structures from compound names
            base_types = sorted({m.split('_')[0] for m in unique_metrics})
            structures = sorted({m.split('_')[1] for m in unique_metrics if '_' in m})

            for b_type in base_types:

                # per-structure bootstrap
                for struct in structures:
                    metric_key = f"{b_type}_{struct}"
                    subset = df_model[df_model['Performance_Metric'] == metric_key]

                    if not subset.empty:
                        metrics_dict = compute_bootstrap_metrics(subset['Value'].values)
                        metrics_dict.update({
                            "Task_ID": task_id,
                            "Model_ID": model_id,
                            "Performance_Metric": b_type,
                            "Region_Target": struct
                        })
                        bootstrap_rows.append(metrics_dict)

                # OVERALL_COMBINED: average across structures per subject, then bootstrap
                global_subset = df_model[df_model['Performance_Metric'].str.startswith(b_type)]
                if not global_subset.empty:
                    subject_means = global_subset.groupby('Subject_ID')['Value'].mean().values
                    metrics_dict = compute_bootstrap_metrics(subject_means)
                    metrics_dict.update({
                        "Task_ID": task_id,
                        "Model_ID": model_id,
                        "Performance_Metric": b_type,
                        "Region_Target": "OVERALL_COMBINED"
                    })
                    bootstrap_rows.append(metrics_dict)

        else:
            # Tasks 1a & 1b detected
            for metric_name in unique_metrics:
                subset = df_model[df_model['Performance_Metric'] == metric_name]

                if not subset.empty:
                    metrics_dict = compute_bootstrap_metrics(subset['Value'].values)
                    metrics_dict.update({
                        "Task_ID": task_id,
                        "Model_ID": model_id,
                        "Performance_Metric": metric_name,
                        "Region_Target": "POPULATION_COHORT"
                    })
                    bootstrap_rows.append(metrics_dict)

    # task-level output in pd df
    output_df = pd.DataFrame(bootstrap_rows)

    # organise columns: ID columns first, then metrics
    column_front = ["Task_ID", "Model_ID", "Performance_Metric", "Region_Target"]
    remaining_cols = [c for c in output_df.columns if c not in column_front]
    output_df = output_df[column_front + remaining_cols]

    # export task bootstrap metrics to CSV
    os.makedirs(OUTPUT_BOOTSTRAP_DIR, exist_ok=True)
    output_csv_path = os.path.join(OUTPUT_BOOTSTRAP_DIR, f"{task_id}_bootstrap_reliability.csv")
    output_df.to_csv(output_csv_path, index=False)
    print(f"Bootstrap metrics for {task_id} exported to: {output_csv_path}")

#########################################################################
##### func: execute this entire py script ###############################
#########################################################################
def main():
    """
    loop through the folder with CSV for each task and run bootstrap publication metrics pipeline for each individual task
    """

    # loop through the data files for each task
    for file_path in task_csv_path_list:
        # load csv of raw metric values for task into pd df
        if not os.path.exists(file_path):
            print(f"Critical Error: Raw metrics CSV missing at path: {file_path}")
            sys.exit(1)
        raw_metric_df = pd.read_csv(file_path)

        # sanity check CSV cols & names
        required_cols = {'Subject_ID', 'Model_ID', 'Task_ID', 'Performance_Metric', 'Value'}
        if not required_cols.issubset(raw_metric_df.columns):
            print(f"Critical Error: Col names in raw metrics CSV at {file_path} don't match required names: {required_cols}")
            sys.exit(1)

        # extract task ID from column in file
        task_id = raw_metric_df['Task_ID'].unique()[0]

        # run bootstrap reliability pipeline on task df
        task_bootstrap_reliability(raw_metric_df, task_id)

if __name__ == "__main__":
    main()