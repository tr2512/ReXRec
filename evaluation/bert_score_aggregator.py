import numpy as np
import pandas as pd
import json

dataset_names = [
    '[Og]Amazon',
    '[Og]Google',
    '[Og]Yelp',
    '[Re]Amazon',
    '[Re]Google',
    '[Re]Yelp'
]
dataset_paths = [
    'amazon',
    'google',
    'yelp',
    're_amazon',
    're_google',
    're_yelp'
]

ablation_names = [
    'XRec'
]
ablation_paths = [
    ''
]
evaluation_data_path = './results/aggregated_bert_scores.json'

evaluation_data = {}

for dataset_name, dataset_path in zip(dataset_names, dataset_paths):
    evaluation_data[dataset_name] = {}

    for ablation_name, ablation_path in zip(ablation_names, ablation_paths):
        evaluation_data[dataset_name][ablation_name] = {}

        df = pd.read_csv(f'./results/bert/BERTScore_{ablation_path}{dataset_path}.csv')

        evaluation_data[dataset_name][ablation_name]['BERTScore_P'] = df['bert_p'].mean()
        evaluation_data[dataset_name][ablation_name]['BERTScore_R'] = df['bert_r'].mean()
        evaluation_data[dataset_name][ablation_name]['BERTScore_F1'] = df['bert_f1'].mean()

        print(dataset_name, ablation_name, evaluation_data[dataset_name][ablation_name])

with open(evaluation_data_path, 'w') as f:
    json.dump(evaluation_data, f, indent=4)