import argparse
from vllm import LLM, SamplingParams
from huggingface_hub import login
import torch
import os
import re
import argparse
import json
import random
from json_repair import repair_json


def postproces(profile, args):
    refined = profile.copy()
    if args.dataset == 'amazon':
        key_name = 'completion'
    else:
        key_name = 'user summary'
    try:
        raw_json = profile['profile']
        fixed_json = repair_json(raw_json)
        fixed_json = json.loads(fixed_json)
        refined[key_name] = fixed_json['summarization']
    except KeyError:
        k = list(fixed_json.keys())
        try:
            assert len(k) == 1
            refined[key_name] = fixed_json[k[0]]
        except:
            refined[key_name] = 'This user profile is not available'
    except TypeError:
        print(raw_json)
        print('-'*20)
        refined[key_name] = 'This user profile is not available'
    del refined['profile']
    return refined


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
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='LLM model')
    parser.add_argument('--num_gpu', type=int, default=2, help='Number of gpu')
    parser.add_argument('--data_folder', type=str, help='Folder contain interactions file')
    parser.add_argument('--batch_size', type=int, default=128, help='Vllm batch size')
    args = parser.parse_args()

    login(args.huggingface_key)

    with open(f'generation/instructions/{args.dataset}_user.txt') as f:
        system_prompt = f.read()
    
    key_mapping = {'amazon': 'completion', 'google': 'business summary', 'yelp': 'business summary'}
    item_profile = {}
    with open(os.path.join(args.data_folder, 'item_profile.json')) as f:
        for line in f:
            d = json.loads(line)
            item_profile[d['iid']] = [d[key_mapping[args.dataset]]]

    key_name = 'title' if args.dataset == 'amazon' else 'name'
    with open(os.path.join(args.data_folder, 'item_content.json')) as f:
        for line in f:
            d = json.loads(line)
            if d['iid'] in item_profile:
                item_profile[d['iid']].append(d['content'][key_name])

    user_information = {}
    with open(os.path.join(args.data_folder, 'interaction.json')) as f:
        for line in f:
            d = json.loads(line)
            if d['item'] in item_profile:
                if d['user'] in user_information:
                    user_information[d['user']].append({key_name: item_profile[d['item']][1], 'description': item_profile[d['item']][0]})
                else:
                    user_information[d['user']] = [{key_name: item_profile[d['item']][1], 'description': item_profile[d['item']][0]}]

    with open(f'generation/instructions/{args.dataset}_user.txt') as f:
        system_prompt = f.read()
    items = []

    if args.dataset == 'amazon':
        prompt_name = 'PURCHASED BOOKS'
    elif args.dataset == 'google':
        prompt_name = 'INTERACTED BUSINESSES'
    elif args.dataset == 'yelp':
        prompt_name = 'PURCHASED BUSINESSES'
    
    for k, v in user_information.items():
        user_prompt = f'{prompt_name}\n{v}'
        prompt = create_prompt(system_prompt, user_prompt, args)
        items.append({'uid': k, 'prompt': prompt})

    idx = 0

    llm = get_llm_model(args)
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
                max_tokens=10000,
                temperature=0.9,
                top_p=1.0,
                stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            )

    with open(os.path.join(args.data_folder, 'user_profile.json'), 'a') as f:
        while idx < len(items):
            batch = items[idx : idx + args.batch_size]
            prompt_data = [i['prompt'] for i in batch]
            raw_responses = llm.generate(prompt_data, sampling_params)
            for response, side_info in zip(raw_responses, batch):
                for out in response.outputs:
                    written = side_info.copy()
                    profile = out.text.strip()
                    del written['prompt']
                    written['profile'] = profile
                    refined_written = postproces(written, args)
                    f.write(json.dumps(refined_written) + '\n')
            if idx == 0:
                print(profile)
            # Move on to the next batch
            idx += args.batch_size

    