# visualization_script_final_right_align_compact.py

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
INPUT_JSON_PATH = './results/aggregated_llm_scores.json' # Verify path
OUTPUT_DIR = './visualisation/ablation/'
OUTPUT_FILENAME = 'comparison_plot.pdf' # New name
TARGET_DATASETS = ['[Og]Amazon', '[Og]Google', '[Og]Yelp']
ABLATION_ORDER = ['XRec', 'XRec (w/o profile)', 'XRec (w/o injection)', 'XRec (w/o profile & injection)']
MODEL_ORDER = ['Gemma2 9b-it', 'Llama 3.1 8b-instruct', 'Llama 3.2 3b-instruct', 'Qwen2.5 7b-instruct']

# --- Styling ---
AGGREGATED_LABEL = "Aggregated"
AGGREGATED_COLOR = 'dimgray'
AGGREGATED_LINESTYLE = '--'
AGGREGATED_MARKER = 'x'
AGGREGATED_LINEWIDTH = 2.0
AGGREGATED_MARKERSIZE = 6

MODEL_MARKERS = ['o', 's', '^', 'D']
MODEL_LINESTYLE = '-'
MODEL_LINEWIDTH = 1.5
MODEL_MARKERSIZE = 5

TITLE_FONTSIZE = 15
AXIS_LABEL_FONTSIZE = 12
TICK_LABEL_FONTSIZE = 10
LEGEND_ITEM_FONTSIZE = 11
LEGEND_TITLE_FONTSIZE = 12
# -----------------------------------------

# 1. Load Data
try:
    with open(INPUT_JSON_PATH, 'r') as f: data = json.load(f)
except FileNotFoundError: print(f"ERROR: JSON file not found at {INPUT_JSON_PATH}"); exit()
except json.JSONDecodeError: print(f"ERROR: Could not decode JSON from {INPUT_JSON_PATH}"); exit()

# 2. Flatten Data
records = []
for dataset, ablations in data.items():
    if dataset in TARGET_DATASETS:
        for ablation, results in ablations.items():
            if ablation in ABLATION_ORDER:
                agg_stats = results.get('aggregated', {})
                if agg_stats: records.append({'Dataset': dataset, 'Ablation': ablation, 'Model': AGGREGATED_LABEL, 'LLMScore': agg_stats.get('LLMScore')})
                per_model = results.get('per_model', {})
                for model, stats in per_model.items():
                    if model in MODEL_ORDER and isinstance(stats, dict): records.append({'Dataset': dataset, 'Ablation': ablation, 'Model': model, 'LLMScore': stats.get('LLMScore')})
df = pd.DataFrame(records)
if df.empty: print("ERROR: No data flattened."); exit()

# 3. Data Preparation
if 'LLMScore' not in df.columns: print("ERROR: Column 'LLMScore' not found."); exit()
df['LLMScore'] = pd.to_numeric(df['LLMScore'], errors='coerce')
df['Ablation'] = pd.Categorical(df['Ablation'], categories=ABLATION_ORDER, ordered=True)
df.dropna(subset=['LLMScore'], inplace=True)
if df.empty: print("ERROR: No valid data after DropNA."); exit()

# 4. Create Plot
num_datasets = len(TARGET_DATASETS)
model_colors = sns.color_palette('tab10', n_colors=len(MODEL_ORDER))
model_color_map = {model: color for model, color in zip(MODEL_ORDER, model_colors)}

sns.set_style("whitegrid")
fig, axes = plt.subplots(1, num_datasets, figsize=(5.5 * num_datasets, 5.0), sharey=True)
if num_datasets == 1: axes = [axes]

all_handles = []
all_labels = []

# 5. Plot Data
print("Generating final plot with right-aligned ticks...")
for i, dataset in enumerate(TARGET_DATASETS):
    ax = axes[i]
    # Plot Aggregated line
    model_or_agg = AGGREGATED_LABEL
    model_df = df[(df['Dataset'] == dataset) & (df['Model'] == model_or_agg)].sort_values('Ablation')
    if not model_df.empty:
        handle = ax.plot(model_df['Ablation'].astype(str), model_df['LLMScore'], marker=AGGREGATED_MARKER, markersize=AGGREGATED_MARKERSIZE, linestyle=AGGREGATED_LINESTYLE, linewidth=AGGREGATED_LINEWIDTH, label=model_or_agg, color=AGGREGATED_COLOR, zorder=10)
        if model_or_agg not in all_labels: all_handles.append(handle[0]); all_labels.append(model_or_agg)
    # Plot Individual Models
    for j, model in enumerate(MODEL_ORDER):
        model_or_agg = model
        model_df = df[(df['Dataset'] == dataset) & (df['Model'] == model_or_agg)].sort_values('Ablation')
        if not model_df.empty:
            handle = ax.plot(model_df['Ablation'].astype(str), model_df['LLMScore'], marker=MODEL_MARKERS[j % len(MODEL_MARKERS)], markersize=MODEL_MARKERSIZE, linestyle=MODEL_LINESTYLE, linewidth=MODEL_LINEWIDTH, label=model_or_agg, color=model_color_map.get(model, 'grey'))
            if model_or_agg not in all_labels: all_handles.append(handle[0]); all_labels.append(model_or_agg)

    # Formatting
    ax.set_title(dataset, fontsize=TITLE_FONTSIZE, weight='bold')

    # *** Set tick labels with RIGHT alignment ***
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE) # Set size separately
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)

    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.grid(axis='x', linestyle=':', alpha=0.4)

    if i == 0:
        ax.set_ylabel("LLMScore", fontsize=AXIS_LABEL_FONTSIZE)

# 6. Create Shared Legend
if all_handles:
    ordered_handles = []; ordered_labels = []
    if AGGREGATED_LABEL in all_labels:
        idx = all_labels.index(AGGREGATED_LABEL)
        ordered_handles.append(all_handles[idx]); ordered_labels.append(all_labels[idx])
    for model_name in MODEL_ORDER:
        if model_name in all_labels:
             idx = all_labels.index(model_name)
             ordered_handles.append(all_handles[idx]); ordered_labels.append(all_labels[idx])

    fig.legend(handles=ordered_handles, labels=ordered_labels,
               title='Model',
               fontsize=LEGEND_ITEM_FONTSIZE,
               title_fontsize=LEGEND_TITLE_FONTSIZE,
               loc='lower center',
               bbox_to_anchor=(0.5, 1.00),
               ncol=len(ordered_handles),
               frameon=False
               )
else: print("Warning: No data plotted, legend not created.")

# 7. Final Adjustments & Save
# Adjust spacing - might need slightly less wspace with ha='right'
# compared to ha='center', but keep bottom margin sufficient
plt.subplots_adjust(
    top=0.84,
    bottom=0.22, # Keep enough space for rotated labels
    left=0.07,
    right=0.98,
    hspace=0.2,
    wspace=0.10 # Can likely tighten wspace slightly with ha='right'
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
try:
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"--- Saved Final plot (Right Align Ticks) to: {output_path} ---")
except Exception as e: print(f"ERROR: Failed to save figure {output_path}. Error: {e}")
plt.close(fig)

print("--- Final visualization script (Right Align Ticks) finished. ---")