import numpy as np
import json
import matplotlib.pyplot as plt

dataset_names  = [
    '[Og]Amazon',
    '[Og]Google',
    '[Og]Yelp'
]
ablation_names = [
    'XRec',
    'XRec (w/o profile)',
    'XRec (w/o injection)',
    'XRec (w/o profile & injection)'
]

llm_evaluation_data_path = './results/aggregated_llm_scores_correct.json'
bert_evaluation_data_path = './results/aggregated_bert_scores.json'

FONT_SIZE = 17
LEGEND_SIZE = 17 + 5
TICK_SIZE = 12
LINE_WIDTH = 1

def add_error_bars(ax, x_pos, mid, err):
    # Plot vertical error bars manually
    ax.vlines(
        x=x_pos, 
        ymin=mid - err, 
        ymax=mid + err,
        colors='black', 
        linestyles='dashed'
    )

    # Add horizontal caps to the error bars
    cap_size = 0.2 
    ax.hlines(
        y=mid - err,
        xmin=x_pos - cap_size,
        xmax=x_pos + cap_size,
        colors='black'
    )

    ax.hlines(
        y=mid + err,
        xmin=x_pos - cap_size,
        xmax=x_pos + cap_size,
        colors='black'
    )


def visualize_ablations(score_ax, stability_ax,
                        evaluation_data, dataset_names, ablation_names,
                        score_name, stability_name, score_lim, stability_lim,
                        legend_flag, error_bars_flag):
    
    spacing_bn_datasets = 2 
    colors = plt.get_cmap('tab10').colors
    hatches = ['/', '\\', 'x', '-', '|'] 

    num_datasets  = len(dataset_names)
    num_ablations = len(ablation_names)

    increment = num_ablations + spacing_bn_datasets
    first_positions = np.arange(num_ablations)
    dataset_positions = [first_positions + increment * i for i in range(num_datasets)]

    score_ax.tick_params(axis='both', labelsize=TICK_SIZE)
    stability_ax.tick_params(axis='both', labelsize=TICK_SIZE)

    for di, dataset_name in enumerate(dataset_names):
        for ai, ablation_name in enumerate(ablation_names):
            # Add score
            score_ax.bar(
                dataset_positions[di][ai], 
                evaluation_data[dataset_name][ablation_name][score_name], 
                width=1,  
                color=colors[ai % len(colors)],  
                # hatch=hatches[ai % len(hatches)],  
                alpha=0.9,  
                edgecolor='black', linewidth=1.5,  
                label=ablation_name if (di == 0 and legend_flag) else ''
            )

            if error_bars_flag:
                add_error_bars(
                    score_ax,
                    dataset_positions[di][ai],
                    evaluation_data[dataset_name][ablation_name][score_name],
                    evaluation_data[dataset_name][ablation_name][score_name + '_std']
                )

            # Add stability
            stability_ax.bar(
                dataset_positions[di][ai], 
                evaluation_data[dataset_name][ablation_name][stability_name], 
                width=1,  
                color=colors[ai % len(colors)],  
                # hatch=hatches[ai % len(hatches)],  
                alpha=0.9,  
                edgecolor='black', linewidth=1.5,  
            )

            if error_bars_flag:
                add_error_bars(
                    stability_ax,
                    dataset_positions[di][ai],
                    evaluation_data[dataset_name][ablation_name][stability_name],
                    evaluation_data[dataset_name][ablation_name][stability_name + '_std']
                )

    score_ax.set_ylabel(score_name + ' →', fontsize=FONT_SIZE, fontweight='bold')
    stability_ax.set_ylabel(stability_name + ' ←', fontsize=FONT_SIZE, fontweight='bold')

    score_ax.set_ylim(score_lim) 
    stability_ax.set_ylim(stability_lim) 

    tick_positions = [np.mean(group) for group in dataset_positions]
    score_ax.set_xticks(tick_positions)
    score_ax.set_xticklabels(dataset_names, fontsize=FONT_SIZE, rotation=20, ha='right', fontweight='bold')
    stability_ax.set_xticks(tick_positions)
    stability_ax.set_xticklabels(dataset_names, fontsize=FONT_SIZE, rotation=20, ha='right', fontweight='bold')

    score_ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=LINE_WIDTH)
    stability_ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=LINE_WIDTH)

with open(llm_evaluation_data_path, 'r') as f:
    llm_evaluation_data = json.load(f)

with open(bert_evaluation_data_path, 'r') as f:
    bert_evaluation_data = json.load(f)

fig, (llmscore_ax, llmstability_ax, bertscore_ax, bertstability_ax) = plt.subplots(nrows=1, ncols=4, figsize=(24, 4))
fig.subplots_adjust(wspace=0.5)  

visualize_ablations(
    llmscore_ax, llmstability_ax, 
    llm_evaluation_data, dataset_names, ablation_names,
    score_name='LLMScore', stability_name='LLMStability',
    score_lim=[40, 85], stability_lim=[5, 18],
    legend_flag=False,
    error_bars_flag=True
)
visualize_ablations(
    bertscore_ax, bertstability_ax,
    bert_evaluation_data, dataset_names, ablation_names,
    score_name='BERTScore_F1', stability_name='BERTStability_F1',
    score_lim=[0.3, 0.45], stability_lim=[0.06, 0.11],
    legend_flag=True,
    error_bars_flag=False
)

fig.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=4, fontsize=LEGEND_SIZE, frameon=True, title_fontsize=FONT_SIZE)
fig.savefig('./visualisation/ablation/ablations.pdf', bbox_inches='tight', dpi=400)
plt.show()