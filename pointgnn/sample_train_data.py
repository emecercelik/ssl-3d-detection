import numpy as np
import argparse
import os

import IPython

def read_index_file(index_filename):
    """Read an index file containing the filenames.
    
    Args:
        index_filename: a string containing the path to an index file.
    
    Returns: a list of filenames.
    """
    
    file_list = []
    with open(index_filename, 'r') as f:
        for line in f:
            file_list.append(line.rstrip('\n'))
    return file_list

def write_indices(path,indices):
    if not os.path.isdir(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
    with open(path,'w') as f:
        for i in indices:
            f.write(i)
            if i != indices[-1]:
                f.write('\n')
    
if __name__ == '__main__':
    '''
    Randomly sample part of the training data split with the given percentage. 
    
    
    Ex: python sample_train_data.py --split_path /kitti_root_3d/pointgnn/3DOP_splits/train.txt --ratio 0.2 --output_name train_20.txt --seed 101
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_path',help='path to the split file')
    parser.add_argument('--ratio', type=float, help='Ratio to sample from the given frames randomly')
    parser.add_argument('--output_name',help='The file name to be saved with the sampled frame indices')
    parser.add_argument('--seed', type=int, default=-1, help='Ratio to sample from the given frames randomly')
    args = parser.parse_args()
    if args.seed != -1:
        np.random.seed(args.seed)
    
    indices = read_index_file(args.split_path)
    
    shuffled_indices = np.random.permutation(indices)
    num_frames = len(indices)
    num_selected_frames = int(args.ratio*num_frames)
    
    selected_indices = shuffled_indices[:num_selected_frames]
    write_indices(os.path.join(os.path.dirname(args.split_path),args.output_name),selected_indices)
    
