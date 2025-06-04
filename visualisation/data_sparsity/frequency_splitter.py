from collections import Counter
import pickle
import numpy as np
import matplotlib.pyplot as plt 
import json_repair
import pandas as pd

def load_sparsity_splits(trn_data, tst_data, tst_ref_explanations):
    trn_uids = trn_data['uid'].tolist()
    trn_uids_counts = Counter(trn_uids)

    tst_uids = tst_data['uid'].tolist()
    tst_explnations = tst_data['explanation'].tolist()

    zero_count_data = [(uid, expl) for uid, expl in zip(tst_uids, tst_explnations) if trn_uids_counts[uid] == 0]
    non_zero_count_data = [(uid, expl) for uid, expl in zip(tst_uids, tst_explnations) if trn_uids_counts[uid] > 0]

    sorted_non_zero_count_data = sorted(non_zero_count_data, key=lambda x: trn_uids_counts[x[0]])

    split_size = len(sorted_non_zero_count_data) // 5 
    boundary_indices = [0] + [split_size * i - 1 for i in range(1, 5)] + [len(sorted_non_zero_count_data) - 1]
    boundary_frequencies = [trn_uids_counts[sorted_non_zero_count_data[b_idx][0]] for b_idx in boundary_indices]

    splits = [zero_count_data]
    split_percentages = [len(zero_count_data)]

    for i in range(len(boundary_frequencies) - 1):
        lower_bound = boundary_frequencies[i]  # Inclusive for the first split, exclusive otherwise
        upper_bound = boundary_frequencies[i + 1]  # Inclusive upper bound

        if i == 0:
            # First split: include [idx 1, idx 2]
            current_split = [
                item for item in sorted_non_zero_count_data
                if lower_bound <= trn_uids_counts[item[0]] <= upper_bound
            ]
        else:
            # Subsequent splits: include (idxn-1, idxn]
            current_split = [
                item for item in sorted_non_zero_count_data
                if lower_bound < trn_uids_counts[item[0]] <= upper_bound
            ]
        splits.append(current_split)
        split_percentages.append(len(current_split))

    # rounding to whole numbers
    split_percentages = np.round(np.divide(split_percentages, len(tst_data)) * 100).astype(int)
    difference = 100 - split_percentages.sum()
    # due to rounding the sum of percentages sometimes gets above 100 so clip that
    if difference != 0:
        split_percentages[np.argmax(split_percentages)] += difference

    sparsity_splits = {}
    for expl in tst_ref_explanations:
        bool = False
        for i, split in enumerate(splits):
            if any(expl == s[1] for s in split):
                bool = True
                sparsity_splits[expl] = i
                break
        assert bool

    return sparsity_splits, split_percentages, boundary_frequencies

def load_dataset(dataset_name):
    with open(f'./data/{dataset_name}/trn.pkl', "rb") as f:
        train_data = pickle.load(f)
    train_data['explanation'] = [(str(exp)).strip() for exp in train_data["explanation"]]
    
    with open(f'./data/{dataset_name}/tst.pkl', "rb") as f:
        test_data = pickle.load(f)
    test_data["explanation"] = [(str(exp)).strip() for exp in test_data["explanation"]]

    with open(f'./data/{dataset_name}/tst_references.pkl', "rb") as f:
        references = pickle.load(f)
    references = [(str(ref)).strip() for ref in references]

    return train_data, test_data, references

def load_results(dataset_name, model_name):
    with open(f'./results/llm/LLMScore_{model_name}_{dataset_name}.txt') as txt:
        json_string = txt.read()

    return json_repair.loads(json_string)

def load_bert_scores(dataset_name):
    return pd.read_csv(f'./results/bert/BERTScore_{dataset_name}.csv')

def aggregate_score(sparsity_splits, result_logs):
    sparsity_set_scores = [[], [], [], [], [], []]
    sparsity_set_means = []
    sparsity_set_stds = []

    for expl, sparsity_set_idx in sparsity_splits.items():
        temp = []
        for results in result_logs:
            temp.append(np.mean(results[expl]["scores"]))
        sparsity_set_scores[sparsity_set_idx].append(np.mean(temp))

    for scores in sparsity_set_scores:
        sparsity_set_means.append(np.mean(scores))
        sparsity_set_stds.append(np.std(scores))

    return sparsity_set_means, sparsity_set_stds

def aggregate_score_bert(sparsity_splits, results):
    sparsity_set_scores = [[], [], [], [], [], []]
    sparsity_set_means = []
    sparsity_set_stds = []


    for expl, sparsity_set_idx in sparsity_splits.items():
        row_dict = results[results["ref"] == expl].iloc[0].to_dict()
        sparsity_set_scores[sparsity_set_idx].append(row_dict["bert_f1"])

    for scores in sparsity_set_scores:
        sparsity_set_means.append(np.mean(scores))
        sparsity_set_stds.append(np.std(scores))

    return sparsity_set_means, sparsity_set_stds

pretty_dataset_names = [
    '[Og]Amazon',
    '[Og]Google',
    '[Og]Yelp'
]
path_dataset_names = [
    'amazon', 
    'google',
    'yelp'
]

re_pretty_dataset_names = [
    '[Re]Amazon',
    '[Re]Google',
    '[Re]Yelp'
]
re_path_dataset_names = [
    're_amazon', 
    're_google',
    're_yelp'
]

path_llm_names = [
    'gemma2',
    'llama3.1',
    'llama3.2',
    'qwen2.5'
]

WIDTH = 25
HEIGHT = 4
FONT_SIZE = 17

def setup_axes(score_ax, stability_ax, score_label, stability_label, score_ylim, stability_ylim):
    x_ticks = ['zero-shot', 'tst1', 'tst2', 'tst3', 'tst4', 'tst5']
    x_positions = np.arange(len(x_ticks))

    for ax in (score_ax, stability_ax):
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_ticks, fontweight='bold', fontsize=FONT_SIZE, rotation=20, ha='right')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    
    score_ax.set_ylim(score_ylim)
    score_ax.set_ylabel(score_label, fontsize=FONT_SIZE, fontweight='bold')
    
    stability_ax.set_ylim(stability_ylim)
    stability_ax.set_ylabel(stability_label, fontsize=FONT_SIZE, fontweight='bold')
    
    return x_positions

def visualise_data_sparsity(
    dataset_names,
    dataset_paths,
    score_ax,
    stability_ax,
    score_label,
    stability_label,
    score_ylim,
    stability_ylim, 
    load_results_func,
    aggregate_func,
    add_labels,
    colors,
    markers
    ):
    x_positions = setup_axes(score_ax, stability_ax, score_label, stability_label, score_ylim, stability_ylim)

    
    for di, (pretty_dataset_name, path_dataset_name) in enumerate(zip(dataset_names, dataset_paths)):

        sparsity_splits, percentages, boundaries = load_sparsity_splits(*load_dataset(path_dataset_name))
     
        print(score_label)
        print(pretty_dataset_name)
        print(percentages)
        print(boundaries)
        print()

        results = load_results_func(path_dataset_name)
        mean_scores, std_scores = aggregate_func(sparsity_splits, results)

        color = colors[di % len(colors)]
        marker = markers[di % len(markers)]

        score_ax.plot(x_positions, mean_scores, label=pretty_dataset_name if add_labels else "", color=color, linestyle='-',
                      marker=marker, markersize=8, markeredgecolor='black')
        
        stability_ax.plot(x_positions, std_scores, color=color, linestyle='-',
                          marker=marker, markersize=8, markeredgecolor='black')

fig_og, (llmscore_ax, llmstability_ax , bertscore_ax, bertstability_ax) = plt.subplots(nrows=1, ncols=4)
fig_og.set_figwidth(WIDTH)
fig_og.set_figheight(HEIGHT)

fig_re, (rellmscore_ax, rellmstability_ax , rebertscore_ax, rebertstability_ax) = plt.subplots(nrows=1, ncols=4)
fig_re.set_figwidth(WIDTH)
fig_re.set_figheight(HEIGHT)

colors = ['red', 'green', 'blue']
markers = ['o', 's', '^']

visualise_data_sparsity(
    pretty_dataset_names,
    path_dataset_names,
    llmscore_ax,
    llmstability_ax,
    'LLMScore →',
    'LLMStability ←',
    [50, 70],
    [9.5, 13.5], 
    lambda path: [load_results(path, llm) for llm in path_llm_names],
    aggregate_score, 
    add_labels=True,
    colors=colors,
    markers=markers
)

visualise_data_sparsity(
    pretty_dataset_names,
    path_dataset_names,
    bertscore_ax,
    bertstability_ax, 
    'BERTScore_F1 →',
    'BERTStability_F1 ←',
    [0.325, 0.475],
    [0.065, 0.105], 
    load_bert_scores,
    aggregate_score_bert,
    add_labels=False,
    colors=colors,
    markers=markers
)

visualise_data_sparsity(
    re_pretty_dataset_names,
    re_path_dataset_names,
    rellmscore_ax,
    rellmstability_ax,
    'LLMScore →',
    'LLMStability ←',
    [50, 70],
    [9.5, 13.5], 
    lambda path: [load_results(path, llm) for llm in path_llm_names],
    aggregate_score,
    add_labels=True,
    colors=colors,
    markers=markers
)

visualise_data_sparsity(
    re_pretty_dataset_names,
    re_path_dataset_names,
    rebertscore_ax,
    rebertstability_ax,
    'BERTScore_F1 →',
    'BERTStability_F1 ←',
    [0.325, 0.475],
    [0.065, 0.105], 
    load_bert_scores,
    aggregate_score_bert,
    add_labels=False,
    colors=colors,
    markers=markers
)
    
fig_og.subplots_adjust(wspace=0.4)
fig_re.subplots_adjust(wspace=0.4)

fig_og.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=6, fontsize=FONT_SIZE+5, frameon=True)
fig_re.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=6, fontsize=FONT_SIZE+5, frameon=True)

fig_og.savefig('./visualisation/data_sparsity/og_data_sparsity.pdf', bbox_inches='tight', dpi=300)
fig_re.savefig('./visualisation/data_sparsity/re_data_sparsity.pdf', bbox_inches='tight', dpi=300)