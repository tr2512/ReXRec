import pickle
import random
import sys
import os
from models.explainer import Explainer
import torch
import argparse
"""
This file runs extra experiments for the XRec reproducibility paper. For more information on running them,
please refer to the README.md file in the root of the repository.

This file contains code for the following tests:
1. Performance of the model ran multiple times on the same user-item pair, evaluated with USR
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_test_embedding(dataset_name):
    with open(f"./data/{dataset_name}/tst.pkl", "rb") as file:
        tst_data = pickle.load(file)
    with open(f"./data/{dataset_name}/user_emb.pkl", "rb") as file:
        user_emb = pickle.load(file)
    with open(f"./data/{dataset_name}/item_emb.pkl", "rb") as file:
        item_emb = pickle.load(file)

    sample = tst_data.sample(n=1).iloc[0]

    sample = sample.to_dict()

    if dataset_name == "amazon":
        system_prompt = "Explain why the user would buy with the book within 50 words."
        item = "book"
    else:
        system_prompt = "Explain why the user would enjoy the business within 50 words."
        item = "business"

    user_message = f"user record: <USER_EMBED> {item} record: <ITEM_EMBED> {item} name: {sample['title']} user profile: {sample['user_summary']} {item} profile: {sample['item_summary']} <EXPLAIN_POS>"
    user_emb_sample = user_emb[sample["uid"]]
    item_emb_sample = item_emb[sample["iid"]]
    prompt = f"<s>[INST] <<SYS>>{system_prompt}<</SYS>>{user_message}[/INST]"
    explanation = sample["explanation"]

    return user_emb_sample, item_emb_sample, prompt, explanation

def generate_expl_test_embedding(user, item, prompt, explanation, dataset, out_name):
    predictions_path = f"{dataset}/{out_name}_pred.pkl"
    references_path = f"{dataset}/{out_name}usr_ref.pkl"

    os.makedirs(dataset, exist_ok=True)

    model = Explainer().to(device)
    user_embedding_converter_path = f"./data/{dataset}/user_converter.pkl"
    item_embedding_converter_path = f"./data/{dataset}/item_converter.pkl"

    model.user_embedding_converter.load_state_dict(
        torch.load(user_embedding_converter_path)
    )
    model.item_embedding_converter.load_state_dict(
        torch.load(item_embedding_converter_path)
    )

    model.eval()
    preds = []
    refs = []

    user = user.to(device)
    item = item.to(device)
    with torch.no_grad():
        for _ in range(100):
            print("Before generate")
            outputs = model.generate(user, item, prompt)
            print(outputs)
            end_idx = outputs[0].find("[")
            if end_idx != -1:
                outputs[0] = outputs[0][:end_idx]

            preds.append(outputs[0])
            refs.append(explanation[0])
    with open(predictions_path, "wb") as file:
        pickle.dump(preds, file)
    with open(references_path, "wb") as file:
        pickle.dump(refs, file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="usr_testing")
    parser.add_argument("--dataset", type=str, default="amazon", help="Dataset name")
    parser.add_argument("--out_name", type=str, default="usr_out", help="Oytput files naming")
    args = parser.parse_args()
    user, item, prompt, explanation = sample_test_embedding(args.dataset)
    generate_expl_test_embedding(user, item, prompt, explanation, args.dataset, args.out_name)
