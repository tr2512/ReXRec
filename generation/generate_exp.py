import argparse
from vllm import LLM, SamplingParams
from huggingface_hub import login
import torch
import os
import re
import argparse
import json
import random


def get_llm_model(args):
    llm = LLM(args.model_name,  
            dtype="half",
            enforce_eager=True,
            gpu_memory_utilization=0.99,
            swap_space=4,
            max_model_len=46500,
            kv_cache_dtype="auto",
            tensor_parallel_size=args.num_gpu
        )
    return llm

def create_prompt(system_prompt, user_prompt, args):
    if args.model_name.split('/')[-1] == 'Llama-3.1-8B-Instruct':
        return f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
    else:
        raise ValueError(f'{args.model_name} model is not supported')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Explanation generation")
    parser.add_argument("--dataset", type=str, default="amazon", help="Dataset name")
    parser.add_argument('--huggingface_key', type=str, help='Huggingface API Key')
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='LLM model')
    parser.add_argument('--num_gpu', type=int, default=2, help='Number of gpu')
    parser.add_argument('--data_folder', type=str, help='Folder contain interactions file')
    parser.add_argument('--num_explanations', type=int, default=100000, help='Number of interactions subsampled')
    parser.add_argument('--batch_size', type=int, default=128, help='Vllm batch size')
    args = parser.parse_args()

    login(args.huggingface_key)

    with open(f'generation/instructions/{args.dataset}_exp.txt') as f:
        system_prompt = f.read()

    interactions = []
    with open(os.path.join(args.data_folder, 'interaction.json')) as f:
        for line in f:
            interaction = json.loads(line)
            if args.dataset != 'yelp':
                if len(interaction['review'].split()) < 10:
                    continue
            user_prompt = {'review': interaction['review']}
            prompt = create_prompt(system_prompt, user_prompt, args)
            interaction['prompt'] = prompt
            interactions.append(interaction)

    random.shuffle(interactions)
    if len(interactions) < args.num_explanations:
        raise ValueError(f'Number of interactions not engough for {args.num_explanations}')
    interactions = interactions[:args.num_explanations]

    idx = 0

    llm = get_llm_model(args)
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
                max_tokens=1000,
                temperature=0.9,
                top_p=1.0,
                stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")], 
            )

    with open(os.path.join(args.data_folder, 'explanation.json'), 'a') as f:
        while idx < len(interactions):
            batch = interactions[idx : idx + args.batch_size]
            prompt_data = [i['prompt'] for i in batch] 
            raw_responses = llm.generate(prompt_data, sampling_params)
            for response, side_info in zip(raw_responses, batch):
                for out in response.outputs:
                    written = side_info.copy()
                    explanation = out.text.strip()
                    written['uid'] = written.pop('user')
                    written['iid'] = written.pop('item')
                    del written['prompt']
                    del written['review']
                    explanation = re.sub(r'^(["\']).*\1$', lambda m: m.group(0)[1:-1], explanation)
                    written['explanation'] = explanation
                    f.write(json.dumps(written) + '\n')
            # Move on to the next batch
            idx += args.batch_size