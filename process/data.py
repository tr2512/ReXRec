'''This file is used to generate data.json file, which will be used in data_handler in explainer.'''
import json
import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os

def combine_profiles_and_explanation(dataset, item_content_path, item_profile_path, \
                                    user_profile_path, explanation_path, data_file_path): 
    attribute_maper = {
        "amazon": {
            "title": "title",
            "item_summary": "completion",
            "user_summary": "completion",
        },
        "yelp": {
            "title": "name",
            "item_summary": "business summary",
            "user_summary": "user summary"
        },
        "google": {
            "title": "name",
            "item_summary": "business summary",
            "user_summary": "user summary"
        }
    }.get(dataset)

    business_title = {}
    with open(item_content_path, "r") as file:
        for line in file:
            process = json.loads(line)
            iid = process["iid"]
            title = process["content"][attribute_maper["title"]]
            business_title[iid] = title

    item_summary = {}
    with open(item_profile_path, "r") as file:
        for line in file:
            process = json.loads(line)
            iid = process["iid"]
            summary = process[attribute_maper["item_summary"]]
            item_summary[iid] = summary

    user_summary = {}
    with open(user_profile_path, "r") as file:
        for line in file:
            process = json.loads(line)
            uid = process["uid"]
            summary = process[attribute_maper["user_summary"]]
            user_summary[uid] = summary

    data = []
    with open(explanation_path, "r") as file:
        for line in file:
            process = json.loads(line)
            uid = process["uid"]
            iid = process["iid"]
            title = business_title[iid]
            user_sum = user_summary[uid]
            item_sum = item_summary[iid]
            explanation = process["explanation"]
            data.append(
                {
                    "uid": uid,
                    "iid": iid,
                    "title": title,
                    "user_summary": user_sum,
                    "item_summary": item_sum,
                    "explanation": explanation,
                }
            )

    # save the data
    with open(data_file_path, "w") as file:
        for d in data:
            file.write(json.dumps(d) + "\n")

def extract_interaction_links(data):
    interaction_links = []

    for datapoint in data:
        interaction_links.append({"user": datapoint["user"], "item": datapoint["item"]})

    return pd.DataFrame(interaction_links)

def extract_interaction_links(data):
    """
    Extracts interaction links from the data.

    Args:
        data (list): List of data points, each containing 'user' and 'item'.

    Returns:
        pd.DataFrame: A DataFrame containing user-item interactions.
    """
    return pd.DataFrame([{ "user": datapoint["uid"], "item": datapoint["iid"] } for datapoint in data])

def split_data(
    total_file_path,
    data_file_path,
    total_trn_csv_file_path,
    total_val_csv_file_path,
    total_tst_csv_file_path,
    trn_pkl_file_path,
    val_pkl_file_path,
    tst_pkl_file_path,
    xpl_trn,
    xpl_val,
    xpl_tst,
    seed
):
    """
    Splits data into training, validation, and test sets and saves the results.

    Args:
        total_file_path (str): Path to the CSV file containing the total dataset.
        data_file_path (str): Path to the JSON data file.
        total_trn_csv_file_path (str): Path to save the training CSV file.
        total_val_csv_file_path (str): Path to save the validation CSV file.
        total_tst_csv_file_path (str): Path to save the test CSV file.
        trn_pkl_file_path (str): Path to save the training pickle file.
        val_pkl_file_path (str): Path to save the validation pickle file.
        tst_pkl_file_path (str): Path to save the test pickle file.
        xpl_trn (int): The amount of explained data allocated to the training set.
        xpl_val (int): The amount of explained data allocated to the validation set.
        xpl_tst (int): The amount of explained data allocated to the test set.
        seed (int): Seed for reproducibility.

    Returns:
        None
    """ 



    # Load total data
    total_data = pd.read_csv(total_file_path)
    
    # Load explained data
    with open(data_file_path, "rb") as file:
        explained_data = [json.loads(line) for line in file]

    # Calculate fractions for splitting
    num_explained_datapoints = xpl_trn + xpl_val + xpl_tst
    assert num_explained_datapoints == len(explained_data), \
        f"xpl_trn ({xpl_trn}) + xpl_val ({xpl_val}) + xpl_tst ({xpl_tst}) not equal to {len(explained_data)} which is the amount of explained data points in {data_file_path}"
    
    num_val_tst = xpl_val + xpl_tst
    VAL_TST_FRAC = num_val_tst / num_explained_datapoints
    TST_FRAC     = xpl_tst / num_val_tst

    # Split explained data into training, validation, and test sets
    explained_trn, explained_val_tst = train_test_split(explained_data, test_size=VAL_TST_FRAC, random_state=seed)
    explained_val, explained_tst = train_test_split(explained_val_tst, test_size=TST_FRAC, random_state=seed)

    # Save the split explained data as pickle files
    for dataset, file_path in zip(
        [explained_trn, explained_val, explained_tst],
        [trn_pkl_file_path, val_pkl_file_path, tst_pkl_file_path]
    ):
        with open(file_path, "wb") as file:
            pickle.dump(pd.DataFrame(dataset), file)

    # Extract interactions from each dataset
    explained_interactions = extract_interaction_links(explained_data)
    explained_trn = extract_interaction_links(explained_trn)
    explained_val = extract_interaction_links(explained_val)
    explained_tst = extract_interaction_links(explained_tst)

    print(f"Explained interactions: {explained_interactions.shape[0]}")
    print(f"Explained interactions -> TRAIN SET: {explained_trn.shape[0]}")
    print(f"Explained interactions -> VALIDATION SET: {explained_val.shape[0]}")
    print(f"Explained interactions -> TEST SET: {explained_tst.shape[0]} \n")

    # Identify unexplained interactions by merging the total interactions with the explained interactions.
    # Retain only rows from the total dataset that do not have a match in the explained interactions.
    merged_data = pd.merge(total_data, explained_interactions, on=["user", "item"], how="left", indicator=True)
    unexplained_interactions = merged_data[merged_data["_merge"] == "left_only"].drop(columns="_merge")

    # Split unexplained interactions into training, validation, and test sets
    unexplained_data = unexplained_interactions.reset_index(drop=True)
    unexplained_trn, unexplained_val_tst = train_test_split(unexplained_data, test_size=VAL_TST_FRAC, random_state=seed)
    unexplained_val, unexplained_tst = train_test_split(unexplained_val_tst, test_size=TST_FRAC, random_state=seed)

    print(f"Unexplained interactions: {unexplained_data.shape[0]}")
    print(f"Unexplained interactions -> TRAIN SET: {unexplained_trn.shape[0]}")
    print(f"Unexplained interactions -> VALIDATION SET: {unexplained_val.shape[0]}")
    print(f"Unexplained interactions -> TEST SET: {unexplained_tst.shape[0]} \n")

    # Combine explained and unexplained interactions
    final_trn_data = pd.concat([explained_trn, unexplained_trn], ignore_index=True)
    final_val_data = pd.concat([explained_val, unexplained_val], ignore_index=True)
    final_tst_data = pd.concat([explained_tst, unexplained_tst], ignore_index=True)

    print(f"Total interactions: {total_data.shape[0]}")
    print(f"Total interactions -> TRAIN SET: {final_trn_data.shape[0]}")
    print(f"Total interactions -> VALIDATION SET: {final_val_data.shape[0]}")
    print(f"Total interactions -> TEST SET: {final_tst_data.shape[0]} \n")

    # Save the final datasets as CSV files
    final_trn_data.to_csv(total_trn_csv_file_path, index=False)
    final_val_data.to_csv(total_val_csv_file_path, index=False)
    final_tst_data.to_csv(total_tst_csv_file_path, index=False)

    # Assert no data leakage
    def assert_no_leakage(df1, df2, name1, name2):
        overlap = pd.merge(df1, df2, on=["user", "item"], how="inner")
        assert overlap.empty, f"Data leakage detected between \'{name1}\' and \'{name2}\': {len(overlap)} overlapping rows."
        print(f"No data leakage found between \'{name1}\' and \'{name2}\' datasets.")

    assert_no_leakage(explained_interactions, unexplained_interactions, "explained", "unexplained")
    
    assert_no_leakage(explained_trn, final_val_data, "training explained", "validation")
    assert_no_leakage(explained_trn, final_tst_data, "training explained", "test")

    assert_no_leakage(explained_val, final_trn_data, "validation explained", "train")
    assert_no_leakage(explained_val, final_tst_data, "validation explained", "test")

    assert_no_leakage(explained_tst, final_trn_data, "validation explained", "train")
    assert_no_leakage(explained_tst, final_val_data, "validation explained", "validation")

def generate_para_dict(total_file_path, total_trn_csv_file_path, total_val_csv_file_path, \
                       total_tst_csv_file_path, para_dict_file_path):
    # Load the final mappings
    df = pd.read_csv(total_file_path)

    # Find the number of unique users and items
    user_num = len(df["user"].unique())
    item_num = len(df["item"].unique())

    # Load the csv files
    data = pd.read_csv(total_trn_csv_file_path)
    trn_user_nb = [[] for _ in range(user_num)]
    trn_item_nb = [[] for _ in range(item_num)]
    for index, row in data.iterrows():
        trn_user_nb[row["user"]].append(row["item"])
        trn_item_nb[row["item"]].append(row["user"])

    data = pd.read_csv(total_val_csv_file_path)
    val_user_nb = [[] for _ in range(user_num)]
    val_item_nb = [[] for _ in range(item_num)]
    for index, row in data.iterrows():
        val_user_nb[row["user"]].append(row["item"])
        val_item_nb[row["item"]].append(row["user"])

    data = pd.read_csv(total_tst_csv_file_path)
    tst_user_nb = [[] for _ in range(user_num)]
    tst_item_nb = [[] for _ in range(item_num)]
    for index, row in data.iterrows():
        tst_user_nb[row["user"]].append(row["item"])
        tst_item_nb[row["item"]].append(row["user"])

    para_dict = {
        "trn_user_nb": trn_user_nb,
        "trn_item_nb": trn_item_nb,
        "val_user_nb": val_user_nb,
        "val_item_nb": val_item_nb,
        "tst_user_nb": tst_user_nb,
        "tst_item_nb": tst_item_nb,
        "user_num": user_num,
        "item_num": item_num,
    }

    with open(para_dict_file_path, "wb") as handle:
        pickle.dump(para_dict, handle)
    print("para_dict saved")

def process(dataset, src_dir, dst_dir, seed,
            xpl_trn, xpl_val, xpl_tst):
    # input
    item_content_path       = os.path.join(src_dir, "item_content.json")

    user_profile_path       = os.path.join(src_dir, "user_profile.json")
    item_profile_path       = os.path.join(src_dir, "item_profile.json")
    explanation_path        = os.path.join(src_dir, "explanation.json")

    total_file_path         = os.path.join(src_dir, "total.csv")

    # output
    data_file_path          = os.path.join(dst_dir, "data.json")

    total_trn_csv_file_path = os.path.join(dst_dir, "total_trn.csv")
    total_val_csv_file_path = os.path.join(dst_dir, "total_val.csv")
    total_tst_csv_file_path = os.path.join(dst_dir, "total_tst.csv")

    trn_pkl_file_path       = os.path.join(dst_dir, "trn.pkl")
    val_pkl_file_path       = os.path.join(dst_dir, "val.pkl")
    tst_pkl_file_path       = os.path.join(dst_dir, "tst.pkl")

    para_dict_file_path = os.path.join(dst_dir, "para_dict.pickle")

    combine_profiles_and_explanation(
        dataset,
        item_content_path,
        item_profile_path,
        user_profile_path,
        explanation_path, 
        data_file_path
    )

    split_data(
        total_file_path,
        data_file_path,
        total_trn_csv_file_path,
        total_val_csv_file_path,
        total_tst_csv_file_path,
        trn_pkl_file_path, 
        val_pkl_file_path,
        tst_pkl_file_path,
        xpl_trn,
        xpl_val,
        xpl_tst,
        seed
    )

    generate_para_dict(
        total_file_path,
        total_trn_csv_file_path,
        total_val_csv_file_path,
        total_tst_csv_file_path,
        para_dict_file_path
    )

if __name__ == '__main__':
    script_description = "This file is used to combine user profile, item profile, and explanation into\
        a single data.json file which is further split into train, validation and test sets."
    
    parser = argparse.ArgumentParser(description=script_description)

    parser.add_argument("--dataset", required=True, choices=["amazon", "yelp", "google"], help="The raw dataset which is to be processed.")
    parser.add_argument("--src_dir", required=True, help="The directory where the item_content, item_profile, user_profile and explanation files are.")
    parser.add_argument("--dst_dir", required=True, help="Directory in which to save the produced files.")
    parser.add_argument("--seed", default=42, type=int, help='Seed used for data splitting.')
    parser.add_argument("--xpl_trn", required=True, type=int, help="The amount of explained data allocated to the training set. Please note that the sum of the training, validation and test sets needs to be equal to the total amount of explained data (the data.json file).")
    parser.add_argument("--xpl_val", required=True, type=int, help="The amount of explained data allocated to the validation set. Please note that the sum of the training, validation and test sets needs to be equal to the total amount of explained data (the data.json file).")
    parser.add_argument("--xpl_tst", required=True, type=int, help="The amount of explained data allocated to the test set. Please note that the sum of the training, validation and test sets needs to be equal to the total amount of explained data (the data.json file).")

    args = parser.parse_args()

    process(
        args.dataset,
        args.src_dir,
        args.dst_dir,
        args.seed,
        args.xpl_trn,
        args.xpl_val,
        args.xpl_tst
    )