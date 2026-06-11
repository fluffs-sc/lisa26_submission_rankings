#!/usr/bin/env python3
"""
Model Submission Ranking and Statistical Testing Pipeline

Input: folder containing separate CSVs for each task, where each CSV is a cleaned long-form CSV of raw performance metric values for a single task, 
Function performs: 1. calc model ranking based on medians & IQR of unweighted geometric rank product of metrics per subject, 2. Wilcoxon statistical testing of the model ranking
Output: Ranked models for a single task in CSV
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
from scipy.stats import rankdata, wilcoxon, iqr
    # scipy.stats.rankdata seems to be the standard choice for ordinal ranking in similar contexts  

#########################################################################
##### global variables ##################################################
#########################################################################

JSON_PATH = "./metric_rank_directions.json"
OUTPUT_SCOREBOARD_DIR = "./final_scoreboards"
RAW_METRICS_FOLDER_PATH = "./task_master_metrics"
task_csv_path_list = glob.glob(f"{RAW_METRICS_FOLDER_PATH}/*.csv")
ALPHA = 0.05

#########################################################################
##### func: load json metric rank direction config ############################
#########################################################################
def load_metric_directions(metric_directions_path):
    """load json file of metric rank directions (for which metric is higher or lower considered a "better" value)."""
    if not os.path.exists(metric_directions_path):
        print(f"Critical error at {metric_directions_path}")
        sys.exit(1)
    with open(metric_directions_path, 'r') as f:
        return json.load(f)

#########################################################################
##### main comprehensive function #######################################
#########################################################################
def task_model_rankings_testing(df_task, task_id, metric_directions):
    """
    For each task
    - Use raw performance metric values to compute ordinal ranks for each model on each performance metric for individual subject scans; 
    - For each model, Use the performance metric ranks corresponding to each subject scan to compute the subject scores for that model by computing an unweighted geometric rank product. 
    - For each model, Use these unweighted geometric rank products (aka individual subject scores for each model) to compute the median score and IQR 
    - median score used as primary ranking metric for leaderboard, IQR seconary ranking metric to break ties
     - Wilcoxon non-para pair-wise statisticaltesting
     """

    ##########################################
    ############### DATA PREP ################
    ##########################################

    ### convert long df to multi-index pd df for easier indexing
        # Rows: subjects of scans & models; 
        # Columns: metrics 
    try:
        matrix = df_task.pivot_table(
            index=['Subject_ID', 'Model_ID'],
            columns='Performance_Metric',
            values='Value'
        )
    except Exception as e:
        print(f"Multi-index data frame conversion error: values must be unique and non-null. Details: {e}")
        return

    ### Extract multi-index pd df into 3 lists of unique values for indexing later and sanity checks
    subjects = matrix.index.get_level_values('Subject_ID').unique().tolist()
    models = matrix.index.get_level_values('Model_ID').unique().tolist()
    metrics = matrix.columns.tolist()
    print(f"Sanity check: # Subjects ={len(subjects)} \n# Models ={len(models)} \n# Metrics ={len(metrics)}")

    ### initialize twoempty 3D numpy array for easiercomputation (one for raw metrics, second for ranks of metrics)
        # axis 0 =rows= models, 
        # axis 1=columns=subject, 
        # axis 2= depth = metrics
    raw_cube = np.zeros((len(models), len(subjects), len(metrics)))
    rank_cube = np.zeros_like(raw_cube)

    # populate one empty 3D numpy array w/ raw metric values from multi-index pd df by iterating through model and subject indices 
    for m_idx, model in enumerate(models):
        for s_idx, subject in enumerate(subjects):
            try:
                raw_cube[m_idx, s_idx, :] = matrix.loc[(subject, model)].values
            except KeyError:
                print(f"Fatal error in populating 3d numpy array: Model '{model}' missing required scores for subject '{subject}'.")
                sys.exit(1)


    ##########################################
    ###### ORDINAL RANKING OF METRICS ########
    ##########################################
    # no metric thresholds implemented for ranking

    ### for each metric for each subject, rank the raw performance values across models
    for s_idx in range(len(subjects)):
        # iterate through metrics
        for metric_idx, metric_name in enumerate(metrics):
            # fetch metric direction from loaded json
            direction = metric_directions.get(metric_name)
            if not direction: # entire metric is missing
                print(f"Error: metric '{metric_name}' missing in metric_rank_directions.json")
                sys.exit(1)

            # extract raw metrics for current subject and metric
            scores_to_rank = raw_cube[:, s_idx, metric_idx]

            # populate empty rank cube with ordinal ranks (scipy.stats.rankdata) of the raw metric values for the current subject and metric using direction rule 
                # scipy.stats.rankdata ranks values from lowest to highest as 1 to N
            if direction == "higher_is_better":
                # multiply raw scoresby -1 so highest (best) score gets ordinal rank 1
                # break ties by averaging (instead of 2 same score models randomly getting ordinal rank n or n+1, they both get n.5)
                rank_cube[:, s_idx, metric_idx] = rankdata(-scores_to_rank, method='average')
            elif direction == "lower_is_better":
                # lowest (best) score default gets ordinal rank 1
                rank_cube[:, s_idx, metric_idx] = rankdata(scores_to_rank, method='average')
            else:
                # catch invalid direction rule for metric existing in json
                print(f"Error: invalid direction rule '{direction}' for metric '{metric_name}' in metric_rank_directions.json")
                sys.exit(1)


    ##########################################
    ## UNWEIGHTED GEOMETRIC RANK PRODUCT for each subject scan for each model ##
    ##########################################
    # Def of unweighted geometric mean = unweighted geometric rank product
        # = (x_1 * x_2 * ... * x_n)^(1/n)
    # Take ln of geometric mean to avoid floating point under&overflow issues
        # = (ln(x_1) + ln(x_2) + ... + ln(x_n))/n

    log_ranks = np.log(rank_cube) # natural log of ordinal ranks
    mean_log_ranks = np.mean(log_ranks, axis=2)  # unweighted geometric average over metric axis, dim: rows=models, cols=subjects
    subject_rank_products = np.exp(mean_log_ranks)  # reverse natural log foreachsubject score for each model, dim: rows=models, cols=subjects


    ##########################################
    ### MEDIAN subject scores & IQR for each model ###
    ##########################################
    model_medians = np.median(subject_rank_products, axis=1) # median of cols=subjects
    model_iqrs = iqr(subject_rank_products, axis=1) # iqr of cols=subjects

    # intermediate df of model medians & iqrs for sanity check
    intermediate_scoreboard_df = pd.DataFrame({
        'Model_ID': models,
        'Median_Rank_Product': model_medians,
        'IQR_Rank_Product': model_iqrs
    })
    # print(intermediate_scoreboard_df)


    ##########################################
    ### ORDERING MODELS BY MEDIAN SCORE & IQR ###
    ##########################################
    # primary: median rank/score (lowermedian rank is better, closer to rank 1)
    # secondary tie-breaker: IQR/ width of middle 50%, low IQR better
    final_scoreboard_df = intermediate_scoreboard_df.sort_values(
        by=['Median_Rank_Product', 'IQR_Volatility'], 
        ascending=[True, True]
    ).reset_index(drop=True)

    ##########################################
    ### WILCOXON SIGNIFICANCE TESTING of MODEL RANKINGS ###
    ##########################################
    # is the difference between sequential model ranks statisticallysignificant across all test subjects or random chance?
    # two-sided signed-rank test, alpha=global var (0.05)
    # null hypothesis: sequential model ranks are identical 

    # initialize column for statistical cluster
    final_scoreboard_df['Statistical_Cluster'] = 1  # default cluster
    current_cluster = 1
    
    # loop through sequential pairs of models 
    for i in range(1, len(final_scoreboard_df)):
        # get the "names" of the models
        model_a_id = final_scoreboard_df.loc[i-1, 'Model_ID']
        model_b_id = final_scoreboard_df.loc[i, 'Model_ID']

        # get the indices of the models using the model "names"
        a_idx = models.index(model_a_id)
        b_idx = models.index(model_b_id)
        
        # get the list of subject rank products for each model 
        vec_a = subject_rank_products[a_idx, :]
        vec_b = subject_rank_products[b_idx, :]

        # wilcoxon two sided signed-rank test
            # manually assign p-values for edge case to avoid division by zero
        if np.array_equal(vec_a, vec_b):
            # manual edge-case handling for exact same ranks of the two models on a single subject
            p_val = 1.0
        else:
            try:
                _, p_val = wilcoxon(vec_a, vec_b, alternative='two-sided')
            except ValueError:
                # manual edge-case handling for when one model's rank is higher/outperforms the other on every single subject
                p_val = 0.0

        # 5% statisticalsignificance threshold
        if p_val < ALPHA:
            # if rank of first model is stat significantly higher than rank of second model, then move second model to one higher statistical cluster/worse rank cluster
            current_cluster += 1
        # if rank of first model is not stat significantly higher than rank of second model, then statistical cluster of second model remains same as first model
        final_scoreboard_df.loc[i, 'Statistical_Cluster'] = current_cluster

    
    # export model ranking df to CSV
    os.makedirs(OUTPUT_SCOREBOARD_DIR, exist_ok=True)
    output_csv_path = os.path.join(OUTPUT_SCOREBOARD_DIR, f"{task_id}_final_scoreboard.csv")
    final_scoreboard_df.to_csv(output_csv_path, index=False)
    print(f"Scoreboard for {task_id} exported to file location: {output_csv_path}")



#########################################################################
##### func: execute this entire py script #######################################
#########################################################################
def main():
    """
    loop through the folder with CSV for each task and run ranking & statistical testing pipeline for each individual task 
    """

    # read in and extract metric rank directions once for all tasks
    metric_config = load_metric_directions(JSON_PATH)
    metric_directions = metric_config["metric_directions"]

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

        # run ranking & statistical testing pipeline on task df
        task_model_rankings_testing(raw_metric_df, task_id, metric_directions)

if __name__ == "__main__":
    main()