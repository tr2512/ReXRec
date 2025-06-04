import pickle
from collections import Counter
import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import List
import numpy as np
from vllm import LLM, SamplingParams
import re
from dataclasses import dataclass
import numpy as np
import evaluate
from sentence_transformers import SentenceTransformer
import argparse
import os

def load_dataset(args):
    try:
        with open(f'data/{args.dataset}/trn.pkl', "rb") as f:
            train_data = pickle.load(f)
        train_data['explanation'] = [(str(exp)).strip() for exp in train_data["explanation"]]
        
        with open(f'data/{args.dataset}/tst.pkl', "rb") as f:
            test_data = pickle.load(f)
        test_data["explanation"] = [(str(exp)).strip() for exp in test_data["explanation"]]

        with open(args.preds, "rb") as f:
            predictions = pickle.load(f)
        predictions = [(str(pred)).strip() for pred in predictions]
        
        with open(args.refs, "rb") as f:
            references = pickle.load(f)
        references = [(str(ref)).strip() for ref in references]

        if len(predictions) != len(references):
            raise ValueError("Mismatch in data size: predictions and references have different lengths.")

        return {
            'trn': train_data,
            'tst': test_data,
            'tst_pred': predictions,
            'tst_ref': references
        }

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def load_sparsity_splits(trn_data, tst_data, tst_ref_explanations):
    trn_uids = trn_data['uid'].tolist()
    trn_uids_counts = Counter(trn_uids)

    tst_uids = tst_data['uid'].tolist()
    tst_explnations = tst_data['explanation'].tolist()

    zero_count_data = [(uid, expl) for uid, expl in zip(tst_uids, tst_explnations) if trn_uids_counts[uid] == 0]
    non_zero_count_data = [(uid, expl) for uid, expl in zip(tst_uids, tst_explnations) if trn_uids_counts[uid] > 0]

    sorted_non_zero_count_data = sorted(non_zero_count_data, key=lambda x: trn_uids_counts[x[0]])

    split_size = len(sorted_non_zero_count_data) // 5 
    splits = [sorted_non_zero_count_data[i*split_size:(i+1)*split_size] for i in range(4)]
    splits.append(sorted_non_zero_count_data[4*split_size:])
    
    splits.insert(0, zero_count_data)

    sparsity_splits = {}
    for expl in tst_ref_explanations:
        for i, split in enumerate(splits):
            if any(expl == s[1] for s in split):
                sparsity_splits[expl] = i
                break
        else:
            print(f'Error: explanation not found in any split')
    
    return sparsity_splits

class BARTScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = 'models/bart.pth'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

    def test(self, batch_size=3):
        """ Test """
        src_list = [
            'This is a very good idea. Although simple, but very insightful.',
            'Can I take a look?',
            'Do not trust him, he is a liar.'
        ]

        tgt_list = [
            "That's stupid.",
            "What's the problem?",
            'He is trustworthy.'
        ]

        print(self.score(src_list, tgt_list, batch_size))

class LLMScoreInference:
    def __init__(self, model_name: str, has_api: bool, api_key: str = None):
        self._model_name = model_name
        self._has_api = has_api
        self._api_key = api_key

        self._model = None
        self._tokenizer = None
        self._sampling_params = None

        if self._has_api:
            self._init_api_model()
        else:
            self._init_vllm_model()

    def _init_api_model(self):
        if not self._api_key:
            raise ValueError("API key must be provided for API-based models.")

        # self._model = OpenAI(api_key=self._api_key)

    def _init_vllm_model(self):
        self._model = LLM(
            model=self._model_name,
            dtype="half",
            enforce_eager=True,
            gpu_memory_utilization=0.99,
            swap_space=4,
            max_model_len=2048,
            kv_cache_dtype="auto",
            tensor_parallel_size=1  # Change this to 1 if you have only 1 GPU
        )
    
        self._tokenizer = self._model.get_tokenizer()
        self._sampling_params = SamplingParams(
            max_tokens=15,
            temperature=0.9,
            top_p=1.0
        )

    def _create_prompts(self, predictions, references, repeat_count=1):
        system_prompt = """Score the given explanation against the ground truth on a scale from 0 to 100, focusing on the alignment of meanings rather than the formatting. Provide your score as a number and do not provide any other text."""
    
        print('Pred', len(predictions))
        print('Ref', len(references))
        prompts = []
        for pred, ref in zip(predictions, references):
            prompt = system_prompt + f"\nPrediction: {pred.strip()}\nReference: {ref.strip()}" + " The score is "
            for _ in range(repeat_count):
                prompts.append(prompt)
        
        return prompts

    def _extract_reference_from_prompt(self, prompt):
        start_keyword = "Reference:"
        end_keyword = "The score is"
    
        start_pos = prompt.find(start_keyword)
        end_pos = prompt.find(end_keyword)
    
        if start_pos == -1 or end_pos == -1 or end_pos <= start_pos:
            # If either keyword is not found or improperly ordered, return None
            return None
    
        reference_text = prompt[start_pos + len(start_keyword):end_pos].strip()
        return reference_text

    def generate(self, result, predictions, references, repeat_count=1, batch_size=1):
        index = 0
        completed = 0
    
        prompts = self._create_prompts(predictions, references, repeat_count)
        total = len(prompts)
    
        while index < len(prompts):
            # Batch processing
            batch = prompts[index: index + batch_size]
            batch_size = len(batch)
            raw_responses = self._model.generate(batch, self._sampling_params)
    
            for response in raw_responses:
                for out in response.outputs:
                    completion = out.text.strip()
                    # Extract the part before the first period or space
                    value_str = re.split(r'[.,\s%/]', completion, maxsplit=1)[0]
                    try:
                        score = float(value_str)

                        if not (0 <= score <= 100):
                            raise ValueError(f"Score {score} is not between 0 and 100.")
                        
                        reference = self._extract_reference_from_prompt(response.prompt.strip())
                        if reference in result:
                            result[reference]["scores"].append(score)
                            completed = completed + 1
                        else :
                            print(response.prompt)
                            print('Nu merge reference finderu')
                    except ValueError:
                        print('Error: Unable to convert to float:', value_str)
                        prompts.append(response.prompt)
    
            index += batch_size
            print(f"Completed so far: {completed} / {total}")
    
        return result


@dataclass
class OthersMetrics:
    bert_precision: float
    bert_precision_std: float
    bert_recall: float 
    bert_recall_std: float
    bert_f1: float
    bert_f1_std: float
    bart_score: float 
    bart_score_std: float
    bleurt: float
    bleurt_std: float
    usr: float
    sts: float
    sts_std: float

class Evaluator: 
    def __init__(self, evaluation_type, model):
        self._evaluation_type = evaluation_type

        if self._evaluation_type == 'LLMScore':
            self._llm_scorer = LLMScoreInference(model_name=model, has_api=False)
        elif self._evaluation_type == 'Others':
            self._bertscore = evaluate.load("bertscore")

            self._bart_scorer = BARTScorer()
            self._bart_scorer.load('evaluation/bart_checkpoint/bart_score.pth')

            self._bleurt_score = evaluate.load("bleurt")

            self._sts_model = SentenceTransformer("all-MiniLM-L6-v2")

    def run_llm_score_with_sparsity(self, predictions, references, sparsity_splits):
        result_template = {ref.strip(): {"scores": [], "quantile": sparsity_splits[ref.strip()]} for ref in references}
        result = self._llm_scorer.generate(result_template, predictions, references, 5, 500)
        
        return result

    def run_bert_score(self, predictions, references, lang="en"):
        results = self._bertscore.compute(
            predictions=predictions, 
            references=references, 
            lang=lang, 
            rescale_with_baseline=True
        )
        
        return np.mean(results["precision"]), np.std(results["precision"]), np.mean(results["recall"]), np.std(results["recall"]), np.mean(results["f1"]), np.std(results["f1"])

    def run_bart_score(self, predictions, references): 
        scores = self._bart_scorer.score(predictions, references, batch_size=100)
        
        return np.mean(scores), np.std(scores)

    def run_bleurt_score(self, predictions, references):
        results = self._bleurt_score.compute(predictions=predictions, references=references)['scores']
    
        return np.mean(results), np.std(results)

    def run_usr_score(self, predictions):
        sequence_batch = [s.split() for s in predictions]
        unique_seq = []

        for seq in sequence_batch:
            if not any((seq == uni_seq) for uni_seq in unique_seq):
                unique_seq.append(seq)
    
        return len(unique_seq) / len(sequence_batch)

    def run_sts_score(self, predictions):
        embeddings = self._sts_model.encode(predictions)

        similarity_matrix = np.inner(embeddings, embeddings)
        similarities = similarity_matrix[np.triu_indices(len(predictions), k=1)]
        
        return np.mean(similarities), np.std(similarities)

    def evaluate(self, predictions, references, sparsity_splits): 
        if self._evaluation_type == 'LLMScore':
            results = self.run_llm_score_with_sparsity(predictions, references, sparsity_splits)
            return results
        elif self._evaluation_type == 'Others':
            # BERTScore Evaluation
            print('--- Starting BERTScore Score ---')
            bert_precision, bert_precision_std, bert_recall, bert_recall_std, bert_f1, bert_f1_std = self.run_bert_score(predictions, references)
    
            # BART Score Evaluation
            print('--- Starting BART Score ---')
            bart_mean, bart_std = self.run_bart_score(predictions, references)
    
            # # BLEURT Score Evaluation
            print('--- Starting BLEURT Score ---')
            bleurt_mean, bleurt_std = self.run_bleurt_score(predictions, references)
    
            # # USR Evaluation 
            print('--- Starting USR Score ---')
            usr = self.run_usr_score(predictions)
    
            # # STS Evaluation
            print('--- Starting STS Score ---')
            sts_mean, sts_std = self.run_sts_score(predictions)

            return OthersMetrics(
                bert_precision=bert_precision,
                bert_precision_std=bert_precision_std,
                bert_recall=bert_recall,
                bert_recall_std=bert_recall_std,
                bert_f1=bert_f1,
                bert_f1_std=bert_f1_std,
                bart_score=bart_mean,
                bart_score_std=bart_std,
                bleurt=bleurt_mean,
                bleurt_std=bleurt_std,
                usr=usr,
                sts=sts_mean,
                sts_std=sts_std
            )

def format_string_others(evaluation):
    return (f""" Evaluation Metrics:
    - BERT Precision: {evaluation.bert_precision} ± {evaluation.bert_precision_std}
    - BERT Recall: {evaluation.bert_recall} ± {evaluation.bert_recall_std}
    - BERT F1: {evaluation.bert_f1} ± {evaluation.bert_f1_std}
    - BART Score: {evaluation.bart_score} ± {evaluation.bart_score_std}
    - BLEURT: {evaluation.bleurt} ± {evaluation.bleurt_std}
    - USR: {evaluation.usr}
    - STS: {evaluation.sts} ± {evaluation.sts_std}
    """)

def save_logs(eval_type, model, dataset_name, logs, args):
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    if eval_type == 'LLMScore':
        file_name = os.path.join(args.out, f"{eval_type}_{model}_{dataset_name}.txt")
        with open(file_name, "a") as f:
            f.write(str(logs))
    elif eval_type == 'Others':
        logs = format_string_others(logs)
        file_name = os.path.join(args.out, f"{eval_type}_{dataset_name}.txt")
        with open(file_name, "a") as f:
            f.write(logs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="explainer")
    parser.add_argument("--dataset", type=str, default="amazon", help="Dataset name")
    parser.add_argument("--preds", type=str, required=True, help='predictions file path')
    parser.add_argument("--refs", type=str, required=True, help='references file path')
    parser.add_argument("--out", type=str, required=True, help='path to save checkpoint logs')
    parser.add_argument("--llm", default=None, help='LLM evaluation model')
    args = parser.parse_args()
    llm_mapping = {
        'llama3.1': 'meta-llama/Llama-3.1-8B-Instruct',
        'llama3.2': 'meta-llama/Llama-3.2-3B-Instruct',
        'gemma2': 'google/gemma-2-9b-it',
        'qwen2.5': 'Qwen/Qwen2.5-7B-Instruct'
    }
    print(os.path.join(args.out, f"LLMScore_{args.llm}_{args.dataset}.txt"))
    dataset = load_dataset(args)
    sparsity_splits = load_sparsity_splits(dataset['trn'], dataset['tst'], dataset['tst_ref'])
    if args.llm is None:
        evaluation_type = 'Others'
        model_name = None
    else:
        evaluation_type = 'LLMScore'
        model_name = llm_mapping[args.llm]
    evaluator = Evaluator(evaluation_type, model_name)
    logs = evaluator.evaluate(dataset['tst_pred'], dataset['tst_ref'], sparsity_splits)
    save_logs(evaluation_type, args.llm, args.dataset, logs, args)


