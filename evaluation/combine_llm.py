import os
import json
import numpy as np
import random
from json_repair import repair_json
import argparse

parser = argparse.ArgumentParser(description="explainer")
parser.add_argument("--out", type=str, required=True, help='path to save checkpoint logs')
args = parser.parse_args()


l = os.listdir(args.out)
l = [i.split('_') for i in l]
results = {}
for i in l:
    path = '_'.join(i)
    dataset = i[-1]
    model = i[1]
    
    if i[0] != 'LLMScore':
        continue

    with open(os.path.join(args.out, path), encoding='utf-8') as f:
        content = f.read()
        content = repair_json(content)
        d = json.loads(content)
        assert len(d) == 3000
        
        for idx, (k, v) in enumerate(d.items()):
            if k in results:
                results[k].append(np.mean(v['scores']))
            else:
                results[k] = [np.mean(v['scores'])]

i = random.choice(list(results.values()))
print(f'Number of LLMs found are {len(i)}')
llm_averages = [np.mean(s) for s in results.values()]
mean = np.mean(llm_averages)
std = np.std(llm_averages)

with open(os.path.join(args.out, 'llmscore.json'), 'w') as f:
    json.dump({'mean': mean, 'std':std}, f) 