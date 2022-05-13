#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from shutil import copyfile
import glob
import IPython

def move_kitti_detection_format(args):
    print('Started!')
    
    new_paths = {'calib':os.path.join(args.output_path,'calib'),
                 'image_2':os.path.join(args.output_path,'image_2'),
                 'label_2':os.path.join(args.output_path,'label_2'),
                 'velodyne':os.path.join(args.output_path,'velodyne') }
    
    # Paths to the tracking dataset
    # the labels should be in KITTI object detection format: 000001.txt instead of <drive_id>.txt
    # 'label_2':os.path.join(args.tracking_path,'data_tracking_label_2/{}/label_02'.format(args.split))
    org_kitti_paths = {'calib':os.path.join(args.tracking_path,'data_tracking_calib/{}/calib'.format(args.split)),
                       'image_2':os.path.join(args.tracking_path,'data_tracking_image_2/{}/image_02'.format(args.split)),
                       'label_2':args.root_dir,
                       'velodyne':os.path.join(args.tracking_path,'data_tracking_velodyne/{}/velodyne'.format(args.split))}
    
    for k in new_paths.keys():
        try:
            os.makedirs(new_paths[k])
            print('{} is generated!'.format(new_paths[k]))
        except:
            print('{} exists!'.format(new_paths[k]))
    
    drives = args.training_drives+args.validation_drives
    drives.sort()
    global_label_counter = 0
    train_img_ids =  []
    val_img_ids = []
    for drive in drives:
        print('Drive {} is being copied!'.format(drive))
        img_addrs = glob.glob(os.path.join(org_kitti_paths['velodyne'],'{:04d}/*.bin'.format(drive)))
        img_ids = [int(os.path.basename(i_ids).split('.')[0]) for i_ids in img_addrs]
        img_ids.sort()
        #drive_label_path = os.path.join(args.root_dir,'{:04d}'.format(drive))
        #path_files = glob.glob(os.path.join(drive_path,'*.txt'))
        #l_labels = len(path_files)
        
        calib_dict = read_calib_file(os.path.join(org_kitti_paths['calib'],'{:04d}.txt'.format(drive)))
            
        for img_id in img_ids:
            img_addr = os.path.join(os.path.dirname(img_addrs[0]),'{:06d}.bin'.format(img_id))
            #img_id = int(os.path.basename(img_addr).split('.')[0])
            prev_img_addr =  os.path.join(os.path.dirname(img_addr),'{:06d}.bin'.format(img_id-1))
            exists_prev_img = prev_img_addr in img_addrs
            # copy labels
            org_label_path = os.path.join(org_kitti_paths['label_2'],'{:04d}/{:06d}.txt'.format(drive,img_id))
            dest_label_path = os.path.join(new_paths['label_2'],'{:06d}.txt'.format(global_label_counter))
            copyfile(org_label_path,dest_label_path)
            # copy images
            org_img_path = os.path.join(org_kitti_paths['image_2'],'{:04d}/{:06d}.png'.format(drive,img_id))
            dest_img_path = os.path.join(new_paths['image_2'],'{:06d}.png'.format(global_label_counter))
            copyfile(org_img_path,dest_img_path)
            # copy velodyne
            org_vel_path = os.path.join(org_kitti_paths['velodyne'],'{:04d}/{:06d}.bin'.format(drive,img_id))
            dest_vel_path = os.path.join(new_paths['velodyne'],'{:06d}.bin'.format(global_label_counter))
            copyfile(org_vel_path,dest_vel_path)
            # copy calib
            write_calib_file(calib_dict,os.path.join(new_paths['calib'],'{:06d}.txt'.format(global_label_counter)))
            
            # trainval split
            if drive in args.training_drives:
                if args.remove_first:
                    if exists_prev_img:
                        train_img_ids.append(global_label_counter)
                    else:
                        print('no prev for {}'.format(img_addr))
                else:
                    train_img_ids.append(global_label_counter)
            else:
                if args.remove_first:
                    if exists_prev_img:
                        val_img_ids.append(global_label_counter)
                    else:
                        print('no prev for {}'.format(img_addr))
                else:
                    val_img_ids.append(global_label_counter)
            global_label_counter+=1
    
    if args.split == 'training':
        write_id_list(os.path.join(args.output_path,'train.txt'),train_img_ids)
        write_id_list(os.path.join(args.output_path,'val.txt'),val_img_ids)
        write_id_list(os.path.join(args.output_path,'trainval.txt'),train_img_ids+val_img_ids)
    else:
        write_id_list(os.path.join(args.output_path,'test.txt'),train_img_ids)
    print('Done!')

def write_id_list(filepath,id_list):
    with open(filepath, 'w') as f:
        for img_id in id_list:
            f.write('{:06d}\n'.format(img_id))

def read_calib_file(filepath):
    ''' Read in a calibration file and parse into a dictionary.
    '''
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            try:
                key, value = line.split(':', 1) # For p0, p1, p2, p3 in tracking
            except:
                key, value = line.split(' ', 1) # For R_rect, Tr_velo_cam, Tr_imu_velo in tracking 
                
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = value #np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def write_calib_file(calib_dict,filepath):
    ''' Write a calibration read into a calib_dict using read_calib_file
    '''
    with open(filepath, 'w') as f:
        f.write('P0:{}\n'.format(calib_dict['P0']))
        f.write('P1:{}\n'.format(calib_dict['P1']))
        f.write('P2:{}\n'.format(calib_dict['P2']))
        f.write('P3:{}\n'.format(calib_dict['P3']))
        try:
            f.write('R0_rect: {}\n'.format(calib_dict['R_rect']))
            f.write('Tr_velo_to_cam: {}\n'.format(calib_dict['Tr_velo_cam']))
            f.write('Tr_imu_to_velo: {}\n'.format(calib_dict['Tr_imu_velo']))
        except:
            f.write('R0_rect: {}\n'.format(calib_dict['R0_rect']))
            f.write('Tr_velo_to_cam: {}\n'.format(calib_dict['Tr_velo_to_cam']))
            f.write('Tr_imu_to_velo: {}\n'.format(calib_dict['Tr_imu_to_velo']))
        


if __name__ == '__main__':
    '''
    Moves files in KITTI Tracking format to KITTI Detection format.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_drives', metavar='N', type=int, nargs='+',help='Drive IDs to be converted and used for training. Use only training_drives if split is testing. (Ex: 11 15 16 18)')
    parser.add_argument('--validation_drives', metavar='N', type=int, nargs='+',help='Drive IDs to be converted and used for validation. Use only training_drives if split is testing. (Ex: 11 15 16 18)')
    parser.add_argument('--root_dir',default=None,help='Path where the drive label folders are kept. Should contain folders 0011, 0015, 0016, 0018 etc. as drive IDs given in the example. The labels should be in KITTI Detection format. Use combine_drive_labels.py to convert.')
    parser.add_argument('--tracking_path',help='Path to the KITTI tracking dataset. Inside: data_tracking_image_2, data_tracking_calib, data_tracking_label_2, data_tracking_velodyne folders')
    parser.add_argument('--output_path',default=None,help='Path showing where the generated labels will be saved. This defines the root path and each drive will be separately saved under a folder with the drive name: 0000/000000.txt,000001.txt,...; 0011/000000.txt,000001.txt,... If None, the same folders are generated under <tracking_path>/drives_in_kitti.')    
    parser.add_argument('--split',default='training',help='training or testing split in KITTI')
    parser.add_argument('--remove_first', action='store_true', help='To skip the first frame of drives in the train.txt and val.txt files')
    args = parser.parse_args()
    
    
    move_kitti_detection_format(args)
