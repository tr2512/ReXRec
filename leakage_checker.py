import pandas as pd
import pickle

def find_leakage(df1, df2):
    overlap = pd.merge(df1, df2, on=["user", "item"], how="inner")
    percentage = int((len(overlap) / len(df1)) * 100)
    return len(overlap), percentage

def extract_interaction_links(data):
    return pd.DataFrame([{ "user": datapoint["uid"], "item": datapoint["iid"] } for datapoint in data])


def detect_leakage(dataset):

    print(f"Detecting leakage between the training, test and validation sets of the \'{dataset}\' dataset \n")

    explained_trn_path = f"data/{dataset}/trn.pkl"
    explained_val_path = f"data/{dataset}/val.pkl"
    explained_tst_path = f"data/{dataset}/tst.pkl"

    total_trn_path = f"data/{dataset}/total_trn.csv"
    total_val_path = f"data/{dataset}/total_val.csv"
    total_tst_path = f"data/{dataset}/total_tst.csv"

    leakage_report_path = f"data/{dataset}/leakage_report.csv"

    with open(explained_trn_path, "rb") as file:
        explained_trn = pickle.load(file)
        explained_trn = explained_trn.rename(columns={'uid': 'user', 'iid': 'item'})

    with open(explained_val_path, "rb") as file:
        explained_val = pickle.load(file)
        explained_val = explained_val.rename(columns={'uid': 'user', 'iid': 'item'})

    with open(explained_tst_path, "rb") as file:
        explained_tst = pickle.load(file)
        explained_tst = explained_tst.rename(columns={'uid': 'user', 'iid': 'item'})

    print(f"Explained training set: {explained_trn.shape[0]} interactions")
    print(f"Explained validation set: {explained_val.shape[0]} interactions")
    print(f"Explained test set: {explained_tst.shape[0]} interactions \n")

    total_trn = pd.read_csv(total_trn_path)
    total_val = pd.read_csv(total_val_path)
    total_tst = pd.read_csv(total_tst_path)

    print(f"Total training set: {total_trn.shape[0]} interactions")
    print(f"Total validation set: {total_val.shape[0]} interactions")
    print(f"Total test set: {total_tst.shape[0]} interactions \n")

    df = pd.DataFrame(columns=["leakage", "percentage_of_set1_in_set2", "set1_name", "set2_name", "set1_path", "set2_path"])


    explained_sets = [explained_trn, explained_val, explained_tst]
    explained_sets_names = ["explanations train", "explanations validation", "explanations test"]
    explained_sets_paths = [explained_trn_path, explained_val_path, explained_tst_path]

    total_sets = [total_trn, total_val, total_tst]
    total_sets_names = ["interactions train", "interactions validation", "interactions test"]
    total_sets_paths = [total_trn_path, total_val_path, total_tst_path]


    df_idx = 0
    for explained_idx in range(len(explained_sets)):
        for total_idx in range(len(total_sets)):
            number, percentage = find_leakage(explained_sets[explained_idx], total_sets[total_idx])
            df.loc[df_idx] = [
                number,
                percentage,
                explained_sets_names[explained_idx],
                total_sets_names[total_idx],
                explained_sets_paths[explained_idx],
                total_sets_paths[total_idx]
            ]
            df_idx += 1

    df.to_csv(leakage_report_path, index=False)
    print(f"Leakage report saved to \'{leakage_report_path}\' \n\n")

detect_leakage("yelp")
detect_leakage("google")
detect_leakage("amazon")

detect_leakage("re_yelp")
detect_leakage("re_google")
detect_leakage("re_amazon")