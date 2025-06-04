'''This file is used to generate total.csv and interaction.json files, which contain the interactions between users and items.'''
import json
import csv
import random
from tqdm import tqdm
from collections import defaultdict
import networkx as nx
import os
import argparse
from attribute_map import get_attribute_map
from metadata import generate_item_content

def collect_useless_item_ids(dataset, metadata_path):
    """Collect IDs of items deemed 'useless' based on dataset-specific criteria."""
    useless_item_ids = set()

    with open(metadata_path, "r") as metadata_file:

        for line in tqdm(metadata_file, desc="Recording item IDs of useless items...", unit=" items"):
            item = json.loads(line)

            if (dataset == "amazon") and (("title" not in item) or ("description" not in item)
                     or (not item["title"]) or (not item["description"])
                     or all(element == "" for element in item["description"])):
                useless_item_ids.add(item["asin"])

            elif (dataset == "yelp") and (("name" not in item) or ("city" not in item) or ("categories" not in item)
                                          or (not item["name"]) or (not item["city"]) or (not item["categories"])):
                useless_item_ids.add(item["business_id"])

            elif (dataset == "google") and (("name" not in item) or ("description" not in item)
                     or (not item["name"]) or (not item["description"])):
                if item["description"]:
                    print(item["description"])
                useless_item_ids.add(item["gmap_id"])

    return useless_item_ids

def filter_reviews(dataset, reviews_path, filtered_interactions_path, useless_item_ids, rating_requirement):
    """Filters out reviews associated to useless items and reviews with less or equal rating than the rating_requirement."""
    user_set = set()
    attribute_map = get_attribute_map(dataset)

    with open(reviews_path, "r") as reviews, open(filtered_interactions_path, "w") as filtered_interactions:
        buffer = []
        buffer_size = 500000

        for line in tqdm(reviews, desc="Filtering reviews based on useless items and rating...", unit=" reviews"):
                    
            review = json.loads(line)
            try:

                if ((review.get(attribute_map['rating'], None) is not None) \
                and (review.get(attribute_map['user'], None) is not None) \
                and (review.get(attribute_map['item'], None) is not None) \
                and (review.get(attribute_map['review'], None) is not None) \
                and (review.get(attribute_map['rating'], 0) > rating_requirement) \
                and (review.get(attribute_map['item'], "") not in useless_item_ids) \
                and (review.get(attribute_map['review'], "") != "")):
                    
                    user_set.add(str(review[attribute_map["user"]]))

                    interaction           = {}
                    interaction["user"]   = str(review[attribute_map["user"]])
                    interaction["item"]   = str(review[attribute_map["item"]])
                    interaction["rating"] = review[attribute_map["rating"]]
                    interaction["time"]   = review[attribute_map["time"]]
                    interaction["review"] = review[attribute_map["review"]]

                    buffer.append(json.dumps(interaction))

                    if len(buffer) >= buffer_size:
                        filtered_interactions.write("\n".join(buffer) + "\n")
                        buffer.clear()

            except:
                print(review)

        if buffer:
            filtered_interactions.write("\n".join(buffer))

    return user_set

def downsample_users(user_ids_set, fraction, seed):
    """Choses a subsample of user ids based on the provided fraction and seed args and returns it."""
    random.seed(seed)

    user_ids = list(user_ids_set)
    user_ids.sort() # Kept for reproducibility as authors did it. Otherwise not needed for sampling.

    sample_size = int(len(user_ids) * fraction)

    return set(random.sample(user_ids, sample_size))

def downsample_interactions(selected_user_ids, filtered_interactions, subsampled_interactions_path):
    """Keeps only the interactions associated to the passed user ids. Removes duplicate interactions."""
    # Remove duplicate interactions and subsample based on users
    with open(filtered_interactions, "r") as filtered_interactions, \
            open(subsampled_interactions_path, mode="w", encoding="utf-8") as sub_interactions_json:
        
        buffer = []
        buffer_size = 500000
        seen_interactions = set()
        items = set()

        for interaction_string in tqdm(filtered_interactions, desc="Downsampling interactions...", unit=" interactions"):
            interaction = json.loads(interaction_string)

            if interaction["user"] in selected_user_ids:
                user_item_pair = (interaction["user"], interaction["item"])

                if user_item_pair not in seen_interactions:
                    items.add(interaction["item"])

                    seen_interactions.add(user_item_pair)
                    
                    buffer.append(json.dumps(interaction))

                    if len(buffer) >= buffer_size:
                        sub_interactions_json.write("\n".join(buffer) + "\n")
                        buffer.clear()
        if buffer:
            sub_interactions_json.write("\n".join(buffer))

        print(f"Number of items after downsampling: {len(items)}")
 
def k_core(subsampled_interaction_path, k_core_interactions_path, tgt_k):
    """Filters interactions such that they satisfy the target k-core property. Returns True if already satisfied, else returns False."""

    G = nx.Graph() # Biggest memory bottleneck 
    
    with open(subsampled_interaction_path, "r") as sub_interactions:

        for interaction_string in tqdm(sub_interactions, desc="Building interaction graph...", unit=" interactions"):
            interaction = json.loads(interaction_string)
            user = interaction["user"]
            item = interaction["item"] + "_item"
            G.add_edge(user, item)

    k_core_graph = nx.k_core(G, tgt_k) # Biggest memory bottleneck 
                             
    with open(subsampled_interaction_path, "r") as sub_interaction, \
        open(k_core_interactions_path, "w") as k_core_interaction:
    
        for interaction_string in tqdm(sub_interaction, desc="Saving k-core interactions", unit=" interactions"):
            interaction = json.loads(interaction_string)

            user = interaction["user"]
            item = interaction["item"] + "_item"

            if (user, item) in k_core_graph.edges():
                k_core_interaction.write(json.dumps(interaction) + "\n")

    return False
                
def remap_ids(temporary_interactions_path, final_interactions_path,
            total_path, user_id_mapping_path, item_id_mapping_path):
    """Remaps the dataset specific user and item ids to sequential integers. Collects dataset statistics."""
    user_id_mapping = defaultdict(lambda: len(user_id_mapping))
    item_id_mapping = defaultdict(lambda: len(item_id_mapping))
    number_of_interactions = 0

    with open(temporary_interactions_path, "r") as temp_interactions, \
        open(final_interactions_path, "w") as interactions, \
        open(total_path, "w") as total_csv:

        csv_writer = csv.writer(total_csv)
        csv_writer.writerow(["user", "item"])

        buffer = []
        buffer_size = 500000

        for interaction_string in tqdm(temp_interactions, desc="Remapping user and item IDs to sequential integers...", unit=" interactions"):
            number_of_interactions += 1
            
            interaction = json.loads(interaction_string)

            original_user_id = interaction['user']
            original_item_id = interaction['item']
            
            remapped_user_id = user_id_mapping[original_user_id]
            remapped_item_id = item_id_mapping[original_item_id]

            csv_writer.writerow([
                remapped_user_id,
                remapped_item_id
            ]) 

            remapped_interaction = {
                "user": remapped_user_id,
                "item": remapped_item_id,
                "rating" : interaction["rating"],
                "time" : interaction["time"],
                "review": interaction["review"]
            }

            buffer.append(json.dumps(remapped_interaction))

            if len(buffer) >= buffer_size:
                interactions.write("\n".join(buffer) + "\n")
                buffer.clear()

        if buffer:
            interactions.write("\n".join(buffer))
            buffer.clear()

    with open(item_id_mapping_path, "w") as f:
        json.dump(dict(item_id_mapping), f)

    with open(user_id_mapping_path, "w") as f:
        json.dump(dict(user_id_mapping), f)

    return len(user_id_mapping), len(item_id_mapping), number_of_interactions

def process(dataset, reviews_path, metadata_path, save_to_dir,
            tgt_k, fraction, seed, required_rating):
    """Data processing pipeline."""
    #----------------------------------------------------------------------------#
    # Intermediate results are saved here. 
    # These files are deleted at the end of this funciton.
    filtered_interactions_path   = os.path.join(save_to_dir, "filtered_interaction.json")
    subsampled_interactions_path = os.path.join(save_to_dir, "subsampled_interaction.json")
    k_core_interactions_path     = os.path.join(save_to_dir, "k_core_interaction.json")

    # User and item ID mappings used in later stages of dataset creation.
    user_id_mapping_path         = os.path.join(save_to_dir, "user_id_mapping.json")
    item_id_mapping_path         = os.path.join(save_to_dir, "item_id_mapping.json")

    # Interactions are saved here.
    interactions_path            = os.path.join(save_to_dir, "interaction.json")
    total_path                   = os.path.join(save_to_dir, "total.csv")

    # Item contents are saved here.
    item_content_path            = os.path.join(save_to_dir, "item_content.json")
    #----------------------------------------------------------------------------#

    useless_item_ids = collect_useless_item_ids(dataset, metadata_path)
    print("Number of items with no title or description:", len(useless_item_ids))

    user_ids_set = filter_reviews(dataset, reviews_path, filtered_interactions_path, useless_item_ids, required_rating)

    print(f"Number of users before downsampling: {len(user_ids_set)}")
    selected_user_ids = downsample_users(user_ids_set, fraction, seed)
    print(f"Number of users after downsampling: {len(selected_user_ids)}")

    downsample_interactions(selected_user_ids, filtered_interactions_path, subsampled_interactions_path)

    is_k_core = k_core(subsampled_interactions_path, k_core_interactions_path, tgt_k)

    if is_k_core:
        previous_step_file = subsampled_interactions_path
    else:
        previous_step_file = k_core_interactions_path
        
    num_users, num_items, num_interactions = remap_ids(
        previous_step_file,
        interactions_path,
        total_path,
        user_id_mapping_path,
        item_id_mapping_path
    )
        
    print("Final number of users: ", num_users)
    print("Final number of items: ", num_items)
    print("Final number of interactions: ", num_interactions)

    # Delete intermediate files which are no longer needed.
    os.remove(filtered_interactions_path)
    os.remove(subsampled_interactions_path)
    if os.path.exists(k_core_interactions_path):
        os.remove(k_core_interactions_path)

    generate_item_content(dataset, item_id_mapping_path, metadata_path, item_content_path)


if __name__ == '__main__':
    script_description = "This file is used to generate total.csv and interaction.json \
        files, which contain the interactions between users and items."
    parser = argparse.ArgumentParser(description=script_description)

    parser.add_argument("--dataset", required=True, choices=["amazon", "yelp", "google"], help="The raw dataset which is to be processed.")
    parser.add_argument("--reviews_path", required=True, help="The path to the reviews file of the raw dataset.")
    parser.add_argument("--metadata_path", required=True, help="The path to the metadata file of the raw dataset.")
    parser.add_argument("--save_to_dir", required=True, help="The path to the directory to which you want to save the produced files.")
    parser.add_argument("--tgt_k", required=True, default=10, type=int, help="Target k (k-core) of the final dataset.")
    parser.add_argument("--fraction", type=float, help="(Optional) Fraction of users to keep (default for all datasets in README).")
    parser.add_argument("--seed", default=42, type=int, help="(Optional) The seed used for choosing the users to keep (default 42).")
    parser.add_argument("--required_rating", default=3, type=int, help="(Optional) Only reviews with rating more than this value are kept (default 3).")

    args = parser.parse_args()

    if (args.fraction == None) or (args.fraction <= 0.0) or (args.fraction > 1.0):
        args.fraction = {
            "amazon": 0.12,
            "yelp": 0.4,
            "google": 0.09
        }.get(args.dataset, None)

        if args.fraction is None:
            # Not going to happen but just in case
            raise ValueError("Invalid dataset")
    
    process(
        args.dataset,
        args.reviews_path,
        args.metadata_path,
        args.save_to_dir,
        args.tgt_k,
        args.fraction,
        args.seed,
        args.required_rating
    )