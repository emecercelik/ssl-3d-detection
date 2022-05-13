from kitti_util import tracking2objdet
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracking_path',help='Path to the KITTI tracking dataset. Inside: data_tracking_image_2, data_tracking_calib, data_tracking_label_2, data_tracking_velodyne folders')
    parser.add_argument('--drive_ids', metavar='N', type=int, nargs='+',help='Drive ids whose labels will be converted. [0,1,2,3]')
    parser.add_argument('--output_path',default=None,help='Path showing where the generated labels will be saved. This defines the root path and each drive will be separately saved under a folder with the drive name: 0000/000000.txt,000001.txt,...; 0011/000000.txt,000001.txt,... If None, the same folders are generated under <tracking_path>/drives_in_kitti.')    
    args = parser.parse_args()
    
    tracking2objdet(args.tracking_path,args.drive_ids,args.output_path)
