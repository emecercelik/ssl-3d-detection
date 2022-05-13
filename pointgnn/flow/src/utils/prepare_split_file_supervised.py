import os

path = "/kitti_root_tracking/supervised_flow/test"
split_name = "val_supervised.txt"
files = os.listdir(path)
npz_files = [f.split('.')[0] for f in files if ".npz" in f]
with open(split_name,'w') as fp:
    write_str = ''
    for npz_f in npz_files:
        write_str+=npz_f+'\n'
    
    fp.write(write_str)
