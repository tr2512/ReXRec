'''This file is used to generate the item_content.json file which contains the metadata of the items'''
import json
from tqdm import tqdm
from attribute_map import get_attribute_map

def extract_item_content(dataset, item):
    if (dataset == "amazon"):
        return {
            "title": item["title"],
            "description": [element for element in item["description"] if element != ""] 
        }
    elif (dataset == "yelp"):
        return {
            "name": item["name"],
            "city": item["city"],
            "categories": item["categories"]
        }
    elif (dataset == "google"):
        return {
            "name": item["name"],
            "description": item["description"]
        }
    
    raise ValueError(f'Unsupported dataset \'{dataset}\'.')

def generate_item_content(dataset, item_id_mapping_path, metadata_path, item_content_path):

    attribute_map = get_attribute_map(dataset)

    item_id_mapping = {}
    with open(item_id_mapping_path, "r") as file:
        item_id_mapping = json.load(file)

    with open(metadata_path, "r") as metadata, open(item_content_path, "w") as item_content:
        buffer = []
        buffer_size = 200000
    
        for line in tqdm(metadata, desc="Generating item content file...", unit=" items"):
            # Parse each line as a JSON object
            item_json = json.loads(line)
            item_id = item_json[attribute_map["item"]]

            if item_id in item_id_mapping:
                prompt = {
                    "iid": item_id_mapping[item_id],
                    "content": extract_item_content(dataset, item_json),
                }

                buffer.append(json.dumps(prompt))

                if len(buffer) >= buffer_size:
                    item_content.write("\n".join(buffer) + "\n")
                    buffer.clear()

        if len(buffer) > 0:
            item_content.write("\n".join(buffer) + "\n")
            buffer.clear()
