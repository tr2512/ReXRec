# aggregation_script.py (Modified)

import numpy as np
import json
import json_repair
import os
from scipy import stats # For confidence intervals

# --- Configuration ---
dataset_names = [
    '[Og]Amazon', '[Og]Google', '[Og]Yelp',
    '[Re]Amazon', '[Re]Google', '[Re]Yelp'
]
dataset_paths = [
    'amazon', 'google', 'yelp',
    're_amazon', 're_google', 're_yelp'
]
ablation_names = [
    'XRec',
    'XRec (w/o profile)',
    'XRec (w/o injection)',
    'XRec (w/o profile & injection)',
    'XRec (w NGCF)',
    'XRec (w/o GNN, fixed)',
    'XRec (w/o GNN, random)',
    'XRec (w/o GNN & MoE)',
]
ablation_paths = [
    '', 'no_profile_', 'no_injection_', 'no_injection_no_profile_',
    'ngcf_', 'random_embeddings_', 'random_random_', 'no_graph_'
]
model_names = [
    'Gemma2 9b-it', 'Llama 3.1 8b-instruct',
    'Llama 3.2 3b-instruct', 'Qwen2.5 7b-instruct'
]
models_paths = ['gemma2', 'llama3.1', 'llama3.2', 'qwen2.5']

output_evaluation_data_path = './results/aggregated_llm_scores.json' # New output name
results_base_dir = './results/llm'

# --- Key name for the list of MEAN scores per output ---
MEAN_SCORES_PER_OUTPUT_KEY = 'mean_scores_per_output'

# --- Helper Function for Stats (Operates on list of MEAN scores per output) ---
def calculate_stats_from_means(mean_scores_list):
    """Calculates LLMScore (mean of means), LLMStability (std of means), N (items), and 95% CI."""
    if not mean_scores_list:
        return {
            'LLMScore': np.nan,       # Mean of means
            'LLMStability': np.nan,   # Std Dev of means
            'N': 0,                   # Number of output items
            'LLMScore_CI_low': np.nan,
            'LLMScore_CI_high': np.nan
        }

    n = len(mean_scores_list) # N is the number of output items
    llm_score = np.mean(mean_scores_list)
    llm_stability = np.std(mean_scores_list) # Std Dev of the means

    if n < 2 or llm_stability == 0: # CI not meaningful
        ci_low = llm_score
        ci_high = llm_score
    else:
        # CI calculation based on the distribution of mean scores
        confidence_level = 0.95
        degrees_freedom = n - 1
        t_crit = np.abs(stats.t.ppf((1 - confidence_level) / 2, degrees_freedom))
        # Standard error OF THE MEAN (of means)
        se = llm_stability / np.sqrt(n) # Use stability (std of means) here
        ci_margin = t_crit * se
        ci_low = llm_score - ci_margin
        ci_high = llm_score + ci_margin

    return {
        'LLMScore': llm_score,
        'LLMStability': llm_stability,
        'N': n,
        'LLMScore_CI_low': ci_low,
        'LLMScore_CI_high': ci_high
    }

# --- Main Processing Logic ---
evaluation_data = {}
print("Starting data aggregation (calculating LLMScore and LLMStability)...")

for dataset_name, dataset_path in zip(dataset_names, dataset_paths):
    print(f"\nProcessing Dataset: {dataset_name}")
    evaluation_data[dataset_name] = {}

    for ablation_name, ablation_path in zip(ablation_names, ablation_paths):
        if dataset_name.startswith('[Re]') and (ablation_name != 'XRec'):
            continue
        print(f"  Processing Ablation: {ablation_name}")

        per_model_results = {}
        all_mean_scores_pooled = [] # Pool MEAN scores per output across models

        for model_name, model_path in zip(model_names, models_paths):
            file_path = os.path.join(results_base_dir, f'LLMScore_{model_path}_{ablation_path}{dataset_path}.txt')

            mean_scores_list_for_model = [] # Store the mean score for each output item
            try:
                with open(file_path, 'r') as f:
                    raw_json_string = f.read()
                    if not raw_json_string.strip():
                        print(f"    WARNING: Empty file skipped: {file_path}")
                        continue
                    try:
                        raw_results = json_repair.loads(raw_json_string)
                    except Exception as e:
                         print(f"    ERROR: Failed to parse JSON in {file_path}: {e}")
                         continue

                    # --- Process Scores Per Output Item ---
                    if isinstance(raw_results, dict):
                        for expl_id, expl_data in raw_results.items():
                            if isinstance(expl_data, dict) and 'scores' in expl_data and isinstance(expl_data['scores'], list):
                                # Filter out non-numeric scores and calculate mean for this output
                                valid_scores = [s for s in expl_data['scores'] if isinstance(s, (int, float)) and not np.isnan(s)]
                                if valid_scores: # Only process if there are valid scores for this output
                                    mean_for_this_output = np.mean(valid_scores)
                                    mean_scores_list_for_model.append(mean_for_this_output)
                                else:
                                    print(f"    WARNING: No valid scores for explanation '{expl_id}' in {file_path}")
                            # else: # Optional: Warn about unexpected structure within an expl_id
                                # print(f"    WARNING: Unexpected structure for explanation '{expl_id}' in {file_path}")
                    else:
                        print(f"    WARNING: Root JSON object in {file_path} is not a dictionary.")

            except FileNotFoundError:
                print(f"    WARNING: File not found, skipping: {file_path}")
                continue
            except Exception as e:
                print(f"    ERROR: Could not read or process file {file_path}: {e}")
                continue

            # --- Calculate Stats from the list of mean scores ---
            if not mean_scores_list_for_model:
                 print(f"    WARNING: No valid mean scores calculated for model '{model_name}' in {file_path}")
                 model_stats = calculate_stats_from_means([])
                 model_stats[MEAN_SCORES_PER_OUTPUT_KEY] = []
            else:
                 model_stats = calculate_stats_from_means(mean_scores_list_for_model)
                 model_stats[MEAN_SCORES_PER_OUTPUT_KEY] = mean_scores_list_for_model # Store the list of means
                 all_mean_scores_pooled.extend(mean_scores_list_for_model) # Add to pooled list

            per_model_results[model_name] = model_stats
            # print(f"      Model '{model_name}': N_items={model_stats['N']}, LLMScore={model_stats.get('LLMScore', 'N/A'):.2f}, LLMStability={model_stats.get('LLMStability', 'N/A'):.2f}") # Debug

        # --- Aggregation across models using pooled MEAN scores ---
        if not all_mean_scores_pooled:
             print(f"  WARNING: No mean scores found for any model for ablation '{ablation_name}'. Skipping aggregation.")
             aggregated_stats = calculate_stats_from_means([])
             if not per_model_results:
                 for model_name in model_names:
                     per_model_results[model_name] = calculate_stats_from_means([])
                     per_model_results[model_name][MEAN_SCORES_PER_OUTPUT_KEY] = []
        else:
             aggregated_stats = calculate_stats_from_means(all_mean_scores_pooled)
             aggregated_stats['N_pooled_items'] = aggregated_stats.pop('N') # Rename N for clarity
             print(f"    Aggregated: N_items={aggregated_stats['N_pooled_items']}, Mean={aggregated_stats.get('LLMScore', 'N/A'):.2f}, Stability={aggregated_stats.get('LLMStability', 'N/A'):.2f}")

        evaluation_data[dataset_name][ablation_name] = {
            "aggregated": aggregated_stats,
            "per_model": per_model_results
        }

# --- Write the final JSON data ---
print(f"\nWriting aggregated data to: {output_evaluation_data_path}")
os.makedirs(os.path.dirname(output_evaluation_data_path), exist_ok=True)
try:
    with open(output_evaluation_data_path, 'w') as f:
        json.dump(evaluation_data, f, indent=4, allow_nan=True) # allow_nan handles potential NaN results
    print("Aggregation finished successfully.")
except Exception as e:
    print(f"ERROR: Failed to write JSON output: {e}")