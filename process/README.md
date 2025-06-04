# README: XRec Data Processing

This README provides detailed instructions for reproducing the datasets used in [re]XRec (if published we can add a link here), based on publicly available data from Amazon, Yelp, and Google. It includes dataset statistics, download links, and commands for processing the data.

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset Details](#dataset-details)
    - [Yelp](#yelp)
    - [Amazon](#amazon)
    - [Google](#google)
3. [XRec Data Processing Pipeline](#xrec-data-processing-pipeline)
4. [Suggestions for Improvement](#suggestions-for-improvement)

---

## Overview
XRec processes raw data from three public datasets to create tailored datasets for recommender system research. The pipeline includes filtering, downsampling, and splitting the data for training, validation, and testing.

---

## Dataset Details

### Yelp
**Original Dataset:**
- Download: [Yelp Dataset](https://www.yelp.com/dataset)
    - `yelp_academic_dataset_review.json`: User reviews.
    - `yelp_academic_dataset_business.json`: Metadata about businesses.
- Statistics:
    - Reviews: **6,990,280**
    - Items: **150,346**

**New Dataset:**
- Download: [Hugging Face](https://huggingface.co/datasets/n-p-petrov/re-x-rec) (TBA)
- Statistics:
    - **10-core interactions**
    - Users: **15,962**
    - Items: **14,085**
    - Interactions: **393,680**

**Commands to Reproduce:**
```bash
python ./process/interaction.py --dataset yelp \
       --reviews_path ./raw_data/yelp/yelp_academic_dataset_review.json \
       --metadata_path ./raw_data/yelp/yelp_academic_dataset_business.json \
       --save_to_dir ./data/re_yelp \
       --tgt_k 10 --seed 42 --fraction 0.4

# Additional commands for Yelp (TBA) -> for generation maybe?

python ./process/data.py --dataset yelp \
       --src_dir ./data/re_yelp/ \
       --dst_dir ./data/re_yelp/ \
       --seed 42 \
       --xpl_trn 74212 \
       --xpl_val 9277 \
       --xpl_tst 3000
```

---

### Amazon

**Original Dataset:**
- Download: [Amazon Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
    - `Books/5-core`: User reviews.
    - `Books/metadata`: Metadata about books.
- Statistics:
    - Reviews: **27,164,983**
    - Items: **2,935,525**

**New Dataset:**
- Download: [Hugging Face](https://huggingface.co/datasets/n-p-petrov/re-x-rec) (TBA)
- Statistics:
    - **10-core interactions**
    - Users: **15,069**
    - Items: **15,028**
    - Interactions: **350,644**

**Commands to Reproduce:**
```bash
python ./process/interaction.py --dataset amazon \
       --reviews_path ./raw_data/amazon/Books_5.json \
       --metadata_path ./raw_data/amazon/meta_Books.json \
       --save_to_dir ./data/re_amazon \
       --tgt_k 10 --seed 42 --fraction 0.12

# Additional commands for Amazon (TBA) -> for generation maybe?

python ./process/data.py --dataset amazon \
       --src_dir ./data/re_amazon/ \
       --dst_dir ./data/re_amazon/ \
       --seed 42 \
       --xpl_trn 95841 \
       --xpl_val 11980 \
       --xpl_tst 3000
```

---
### Google
**Original Dataset:**
- Download: [Google Review Dataset](https://jiachengli1995.github.io/google/index.html)
    - `California/10-core reviews`: User reviews.
    - `California/metadata`: Metadata about items.
- Statistics:
    - Reviews: **44,476,890**
    - Items: **515,961**

**New Dataset:**
- Download: [Hugging Face](https://huggingface.co/datasets/n-p-petrov/re-x-rec) (TBA)
- Statistics:
    - **10-core interactions**
    - Users: **19,503**
    - Items: **18,998**
    - Interactions: **400,038**

**Commands to Reproduce:**
```bash
python ./process/interaction.py --dataset google \
       --reviews_path ./raw_data/google/review-California_10.json \
       --metadata_path ./raw_data/google/meta-California.json \
       --save_to_dir ./data/re_google \
       --tgt_k 10 --seed 42 --fraction 0.09

# Additional commands for Google (TBA) -> for generation maybe?

python ./process/data.py --dataset google \
       --src_dir ./data/re_google/ \
       --dst_dir ./data/re_google/ \
       --seed 42 \
       --xpl_trn 94663 \
       --xpl_val 11833 \
       --xpl_tst 3000
```

---

## XRec Data Processing Pipeline

1. **`interaction.py`**
    - Filters items without a name or description.
    - Filters reviews for items lacking a name or description.
    - Excludes reviews with ratings ≤ 3 (configurable with `--required_rating`).
    - Downsamples interactions using a fraction (configurable with `--fraction`).
    - Ensures a 10-core interaction graph (configurable with `--tgt_k`).
    - Remaps user and item IDs to sequential integers.
    - Extracts item name, description, and categories.

2. **Data Generation**
    - **(TBA)** Additional details about data generation commands?

3. **`data.py`**
    - Combines item profiles, user profiles, and ground truth explanations.
    - Splits data into training, validation, and test sets.
    - Prepares neighborhood information for collaborative filtering.
---