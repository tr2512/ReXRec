<div align="center">
<h3> Revisiting XRec: How Collaborative Signals Influence LLM-Based Recommendation Explanations </h3>

**Authors:** [Catalin E. Brita](https://github.com/C-Brita)\*, [Hieu Nguyen](https://github.com/tr2512)\*, [Lubov Chalakova](https://github.com/LubovCh)\*  [Nikola Petrov](https://github.com/n-p-petrov)\*,

</div>

\* Equal contribution.

[TMLR Paper](https://openreview.net/forum?id=cPtqOkxQqH)

## Abstract
Recommender systems help users navigate large volumes of online content by offering personalized recommendations. However, the increasing reliance on deep learning-based techniques has made these systems opaque and difficult to interpret. To address this, XRec [1] was introduced as a novel framework that integrates collaborative signals and textual descriptions of past interactions into Large Language Models (LLMs) to generate natural language explanations for recommendations. In this work, we reproduce and expand upon the findings of XRec [1]. While our results validate most of the original authors’ claims, we were unable to fully replicate the reported performance improvements from injecting collaborative information into every LLM attention layer, nor the claimed effects of data sparsity. Beyond replication, our contributions provide evidence that the Graph Neural Network (GNN) component does not enhance explainability. Instead, the observed performance improvement is attributed to the Collaborative Information Adapter, which can act as a form of soft prompting, efficiently encoding task-specific information. This finding aligns with prior research suggesting that lightweight adaptation mechanisms can condition frozen LLMs for specific downstream tasks. Our implementation is open-source.

1. Qiyao Ma, Xubin Ren, and Chao Huang. Xrec: Large language models for explainable recommendation. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 391–402, Miami, Florida, USA, 2024. Association for Computational Linguistics. 
## Environment setup

In this project, we use two different environments, one for training and inference, others for reproducing dataset and evaluation due to version incompatible between the two process (training use huggingface transformers, evaluation use vllm). Both use python 3.10.11
### Creating training environments
```
python3 -m venv training_environment
source training_environment/bin/activate
pip install -r requirements.txt
```
### Creating evaluation environments
```
python3 -m venv evaluation_environment
source evaluation_environment/bin/activate
pip install -r second_requirements.txt
mkdir evaluation/bart_checkpoint
gdown 1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m
cp bart_score.pth evaluation/bart_checkpoint
```

## Reproduce the dataset

The following steps describe how to reproduce the dataest.

To reproduce the dataset, you need to follow the following four steps:
1. Download the original full dataset
2. Subsample the original dataset by first filtering for users with at least k interactions, then randomly select a specified fraction of those qualifying users.
3. Generate the groundtruth explanations, user profiles and item profiles from the subsampled dataset.
4.  Split the data into training/validating/testing.

### Activate the environment
For this section, evaluation environment is required
```
deactivate
source evaluation_environment/bin/activate
```
### Download full dataset:
-  Yelp dataset: https://www.yelp.com/dataset. Put under raw_data/yelp
-  Amazon dataset: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/. Download the book dataset, 5-core version. Put the dataset under raw_data/amazon
-  Google dataset: https://jiachengli1995.github.io/google/index.html. Download California dataset, 10 cores version including reviews and metadata. Put the dataset under raw_data/google

### To subsamples the dataset:

```
python ./process/interaction.py --dataset <dataset_name> \
       --reviews_path <reviews_filepath> \
       --metadata_path <metadata_filepath> \
       --save_to_dir <save_dir> \
       --tgt_k <core_number> --seed <seed> --fraction <user_fractions>
```

where:
- dataset: Name of the dataset: amazon, google and yelp
- reviews_path: Location of the dataset reviews.
    - amazon: ./raw_data/amazon/Books_5.json
    - google: ./raw_data/google/review-California_10.json
    - yelp: ./raw_data/yelp/yelp_academic_dataset_review.json
- metadata_path: Location of the dataset metadata
    - amazon: ./raw_data/amazon/meta_Books.json
    - google: ./raw_data/google/meta-California.json
    - yelp: ./raw_data/yelp/yelp_academic_dataset_business.json
- save_to_dir: The new dataset saved path
- tgt_k: Number of core: our reproduce works used 10
- seed: Seed
- fraction: Fraction of users subsample. Our reproduced work used 0.12, 0.09 and 0.4 for amazon, google and yelp respectively


### To generate groundtruth explanation, user and item profile:

```
# Generate groundtruth explanation
python generation/generate_exp.py --dataset <dataset> \
        --model_name <model_name> \
        --huggingface_key <huggingface_key> \
        --num_gpu <num_gpu> \
        --data_folder <data_folder> \
        --num_explanations <num_explanations> \
        --batch_size <batch_size>

# Generate item profile
python generation/generate_item_profile.py --dataset <dataset> \
        --model_name <model_name> \
        --huggingface_key <huggingface_key> \
        --num_gpu <num_gpu> \
        --data_folder <data_folder> \
        --batch_size <batch_size>

# Generate user profile
python generation/generate_user_profile.py --dataset <dataest> \
        --model_name <model_name> \
        --huggingface_key <huggingface_key> \
        --num_gpu <num_gpu> \
        --data_folder <data_folder> \
        --batch_size <batch_size>
```
where:
- dataset: dataset to generate: amazon, google, yelp
- model_name: model use to generate generations. Current version only supprots llama3.1-8B: meta-llama/Llama-3.1-8B-Instruct
- huggingface_key: huggingface key of the account that has access to the model used.
- num_gpu: number of gpu 
- data_folder: new dataset path
- num_explanations: number of explanations generated. For matching with datasets in the original work, we use 110821, 109496 and 86489 for amazon, google and yelp respectively
- batch_size: batch size

### Train/val/test split
```
python ./process/data.py --dataset <dataset> \
       --src_dir <src_dir> \
       --dst_dir <tst_dir> \
       --seed <seed> \
       --xpl_trn <num_train> \
       --xpl_val <num_val> \
       --xpl_tst <num_test>
```
- dataset: dataset name: amazon, google and yelp
- src_dir: location of the dataset
- tgt_dir: location to create train val and test file
- seed: seed for data splitting
- num_train: number of training samples 
- num_val: number of validation samples
- num_test: number of test samples 

For reproduce the dataset, number of train/val/test data are:
- yelp: 74212/9277/3000
- amazon: 95841/11980/3000
- google: 94663/11833/3000
## Train the GNN
For this section, training environment is used:
```
deactivate
source training_environment/bin/activate
```
To train the GNN recommender system model:
```
python encoder/train.py --dataset <dataset> --model <model>
```
- dataset: dataset to train the gnn on: amazon, yelp, google, re_amazon, re_yelp, re_google (re mean reproduce dataset).
- model: GNN model: ngcf, light-gcn
After training the user and item embeddings will be saved at the data folder with the name ```user_embeds_<model>.pkl``` and ```item_embeds_<model>.pkl```
## Train the explainer model
For this section, training environment is used:
```
deactivate
source training_environment/bin/activate
```
To train the explainer model we use the command
```
python explainer/main.py --mode finetune \
--dataset <dataset> 
--graph <graph> \
--out_name full \
--no-profile \
--no-injection \
--random-embeddings \
--random-random 
```
- mode: training or inference. In this section for training choose finetune
- dataset: dataset to train on: amazon, google, yelp, re_amazon, re_google, re_yelp
- graph: gnn embeddings to use: light-gcn, ngcf
- out_name: name of the experiments
- no-profile (optional): choose this option if you want to train without user and item profile
- no-injection (optional): choose this option if you want to train without injection
- --ramdom-embeddings(optional): initialize unique random embeddings for each user and item
- --random-random(optional): initialize random embeddings for users and items that change every iteration

After training, the checkpoints will be saved at the data folder with name ```user_converter_<out_name>.pkl``` and ```item_converter_<out_name>.pkl```. 
## Model inference
For inference, use the exact same command as above, but change finetune to generate. The predictions and references will be saved to the data folder with the name ```tst_predictions_<out_name>.pkl``` and ```tst_references_<out_name>.pkl``` 
## Evaluation
For this section, evaluation environment is used:
```
deactivate
source evaluation_environment/bin/activate
```
```
python evaluation/main.py --dataset <dataset> \
        --preds <predictions_file_path> \
        --refs <references_file_path> \
        --out <evaluation_logs_path> \
        --llm <llm_name> 
```
Parameters:
- dataset: dataset name (amazon, google, yelp, re_amazon, re_google, re_yelp)
- preds: filepath of the predictions file
- refs: filepath of references path
- out: path to save evaluation logs
- llm (optional): llm model use to evaluate: qwen2.5, llama3.1, llama3.2, gemma2. If this is not selected, the evaluation code will instead return BARTScore, BERTScore, BLEURT, USR and STS.

Each llm evaluation will return a log file of that llm score. To average 4 llm scores into a single score, run the following code
```
python evaluation/combine_llm.py --out test_dir
```
test_dir: path of evaluation logs

## Logs
Evaluation logs of LLMScore and BERTScore is located under ```results``` folder.

## Checkpoints 
We provide the reproduced dataest and the checkpoints [here](https://drive.google.com/drive/folders/1WBPHekWq37AHhdyO3EL2dmMw9NIGBcWc?usp=sharing). The folders including:
- reproduce_dataset: reproduce version of amazon, google and yelp dataset.
- GNN_embedding: the GNN embeddings trained with LightGCN, NGCF and randomly initialized.
- checkpoint: MoE checkpoint (user and item converter) of the main result and different ablations.
- predictions: model inference results on the main model and different ablations
## Acknowledgement
The code is heavily adapted from [XRec](https://github.com/HKUDS/XRec)