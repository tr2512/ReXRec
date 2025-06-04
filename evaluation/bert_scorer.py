import numpy as np
import evaluate
import pickle
import pandas as pd
from tqdm import tqdm

def BERT_score(predictions, references):
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        rescale_with_baseline=True,
    )
    return results

dataset_names = ['amazon', 'google', 'yelp', 're_amazon', 're_google', 're_yelp']

for dataset_name in dataset_names:
    # Load predictions
    print(dataset_name)
    with open(f'./data/{dataset_name}/tst_predictions.pkl', "rb") as f:
        predictions = pickle.load(f)
    predictions = [(str(pred)).strip() for pred in predictions] 

    # Load references
    with open(f'./data/{dataset_name}/tst_references.pkl', "rb") as f:
        references = pickle.load(f)
    references = [(str(ref)).strip() for ref in references]

    # Compute BERT scores
    bert_scores = BERT_score(predictions, references)

    # Create a list of dictionaries for DataFrame
    dict_list = []
    for i, (pred, ref) in enumerate(tqdm(zip(predictions, references), total=len(predictions))):
        entry = {
            "pred": pred,
            "ref": ref,
            "bert_p": bert_scores["precision"][i],
            "bert_r": bert_scores["recall"][i],
            "bert_f1": bert_scores["f1"][i]
        }
        dict_list.append(entry)

    df = pd.DataFrame(dict_list)
    path = f'./results/bert/{dataset_name}_tst_bert_scores.csv'
    print(f'Saving bert scores to: {path}')
    df.to_csv(path, index=False)
