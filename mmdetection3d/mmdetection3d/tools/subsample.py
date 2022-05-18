import pickle
import sys
import random
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--input', dest='f_path', type=str, required=True)
parser.add_argument('--output', dest='new_path', type=str, required=False)
parser.add_argument('--ratio', dest='ratio', type=float, required=True)
parser.add_argument('--pairwise', dest='pairwise', type=ast.literal_eval, default=False)
args = parser.parse_args()

f_path = args.f_path
ratio = args.ratio
pairwise = args.pairwise
print(f'Subsampling {f_path} with ratio {ratio}.')
with open(f_path, 'rb') as f:
    data = pickle.load(f)
l = len(data['infos'])
rand_idx = []
if pairwise:
    even_rand_idx = random.sample([i for i in range(0, l, 2)], k = int(l * ratio // 2))
    for i in even_rand_idx:
        rand_idx.extend([i, i + 1])
else:
    rand_idx = random.sample([i for i in range(l)], k = int(l * ratio))
rand_idx.sort()
new_infos = [data['infos'][i] for i in rand_idx]
new_data = {'infos': new_infos, 'metadata': data['metadata']}
if args.new_path is None:
    if pairwise:
        new_path = f_path[:f_path.rfind('.')] + f'_{ratio}_pairwise' + f_path[f_path.rfind('.'):]
    else:
        new_path = f_path[:f_path.rfind('.')] + f'_{ratio}' + f_path[f_path.rfind('.'):]
else:
    new_path = args.new_path
print('Sampled {} samples out of {}.'.format(len(new_infos), l))
print(f'Writing into {new_path}.')
with open(new_path, 'wb') as fw:
    pickle.dump(new_data, fw)
print('Done.')

