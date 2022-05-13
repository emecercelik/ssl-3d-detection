#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image,ImageDraw,ImageFont
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from lib.misc import show_image,tf_iou
import tensorflow as tf
import IPython

#### To do the evaluation and extract features
line_feat = ['frame','track_id','type','truncated','occluded','alpha', 'x1_2d','y1_2d',
             'x2_2d','y2_2d','h_3d','w_3d','l_3d','x_3d','y_3d','z_3d','rotation_y','score']

object_types = ['Car','Van','Truck','Pedestrian','Person_sitting','Cyclist','Tram','Misc','DontCare']

class obj_detection_dataset():
    def __init__(self,image_path=None,label_path=None,test_data=False):
        """
        The image files and the object ground truth files are named with same indices. So a label file 
        having the same index with an image file contains the ground-truth objects in that image. Each
        has only one object and its features. 
        """
        
        self.image_path = image_path
        self.label_path = label_path
        
        self.image_addresses = glob.glob(self.image_path)
        self.image_addresses.sort()
        
        self.len_addr =  len(self.image_addresses)
        self.test_data = test_data
        self.type_dict = {'frame':int,'track_id':int,'type':str,'truncated':int,'occluded':int,
                          'alpha':float, 'x1_2d':float,'y1_2d':float,'x2_2d':float,'y2_2d':float,
                          'h_3d':float,'w_3d':float,'l_3d':float,'x_3d':float,'y_3d':float,'z_3d':float,
                          'rotation_y':float}
        
        self.type_dict_test = {'frame':int,'track_id':int,'type':str,'truncated':int,'occluded':int,
                              'alpha':float, 'x1_2d':float,'y1_2d':float,'x2_2d':float,'y2_2d':float,
                              'h_3d':float,'w_3d':float,'l_3d':float,'x_3d':float,'y_3d':float,'z_3d':float,
                              'rotation_y':float}
        self.line_feat = ['frame','track_id','type','truncated','occluded','alpha', 'x1_2d','y1_2d',
             'x2_2d','y2_2d','h_3d','w_3d','l_3d','x_3d','y_3d','z_3d','rotation_y']
        #self.var_types = [int,int,str,int,int,float,float,float,float,float,float,float,float,float,float,float,float]
        self.line_feat_test = ['frame','track_id','type','truncated','occluded','alpha', 'x1_2d','y1_2d',
             'x2_2d','y2_2d','h_3d','w_3d','l_3d','x_3d','y_3d','z_3d','rotation_y','score']
        #self.var_types_test = [int,int,str,int,int,float,float,float,float,float,float,float,float,float,float,float,
        #                       float,float]
        self.read_labels_track()
        
    def read_labels_track(self):
        """
        Reads tracking label txt files of the provided KITTI tracking video using the provided label path. 
        
        Returns obj_dict and obj_list
            obj_dict: A dictionary that contain objects of each frame as a separate dictionary denoted by an integer frame id
                {0:{o1,o2,o3,...},1:{o1,o2,o3,...},...}
                o1 = {'frame':0,'track_id':0,'type':"Car",...}
                o2 = {'frame':1,'track_id':2,'type':"Pedestrian",...}
            obj_list: A list that contains objects of each frame. Objects are the object_kitti_tracking instances
                [[o1,o2,o3,...],[o1,o2,o3,...],...]
        """
        f = open(self.label_path,"r")
        lines = f.readlines() 
        obj_dict = {}
        obj_list = {}
        for line in lines:
            if self.test_data:
                data_one_line = line.split(" ")
                one_line_dict ={}
                for dat,feat in zip(data_one_line,self.line_feat_test):
                    one_line_dict[feat] = self.type_dict_test[feat](dat)
                
                obj = object_kitti_tracking(one_line_dict)
                
                try:
                    obj_dict[one_line_dict['frame']].append(one_line_dict)
                    obj_list[one_line_dict['frame']].append(obj)
                except Exception as e:
                    obj_dict[one_line_dict['frame']] = [one_line_dict]
                    obj_list[one_line_dict['frame']] = [obj]
                    #obj_list.append([obj])
            else:
                data_one_line = line.split(" ")
                one_line_dict ={}
                for dat,feat in zip(data_one_line,self.line_feat):

                    try:
                        one_line_dict[feat] = self.type_dict[feat](dat)
                    except:
                        one_line_dict[feat] = self.type_dict[feat](float(dat))
                
                obj = object_kitti_tracking(one_line_dict)
                try:
                    obj_dict[one_line_dict['frame']].append(one_line_dict)
                    obj_list[one_line_dict['frame']].append(obj)
                except Exception as e:
                    obj_dict[one_line_dict['frame']] = [one_line_dict]
                    obj_list[one_line_dict['frame']] = [obj]
                    #obj_list.append([obj])
        
        f.close()
        self.obj_dict,self.obj_list = obj_dict,obj_list
        return obj_dict,obj_list

    
    def get_image_2(self,ind,resize=None):
        """
        Reads and resizes image using PIL.Image library. 
        
        Returns the image (resized if resize is not None) and its original shape
        """
        im = Image.open(self.image_addresses[ind]).convert("RGB")
        real_shape = np.shape(im)
        if resize is not None:
            im = im.resize(resize, Image.ANTIALIAS)
        
        return np.array(im),real_shape

    def get_images(self,ind_list,resize=None):
        """
        returns images,shapes 
        shapes is the list of real shapes of images after reshaping into resize dimensions
        """
        images = []
        shapes = []
        for ind in ind_list:
            img,real_shape = self.get_image_2(ind,resize)
            images.append(img)
            shapes.append(real_shape)
        return images,shapes
    
    def check_object_type(self,obj,val_type):
        '''
        Checks whether given kitti object instance fits into the data validation type ('easy','moderate','hard')
        
        obj : kitti object instance
        val_type: 'easy', 'moderate', 'hard'
        
        returns true or false 
        '''
        
        if val_type == 'easy':
            if np.abs(obj.y1-obj.y2)>=40 and obj.truncated<=0.15 and (obj.occluded in [0,3]):
                return True
            else:
                return False
                
        elif val_type == 'moderate':
            if np.abs(obj.y1-obj.y2)>=25 and obj.truncated<=0.3 and (obj.occluded in [0,1,3]):
                return True
            else:
                return False
        elif val_type == 'hard':
            if np.abs(obj.y1-obj.y2)>=25 and obj.truncated<=0.5:
                return True
            else:
                return False
        else:
            return False
        
    def show_image(self,ind,ground_truth=False,box_proposals=None,save=None,
                   filter_out_classes=['Van','Truck','Person_sitting','Tram','Misc','DontCare'],
                  print_class_name=False,val_type=None,resize=None):
        """
        ind: index of the image to be read from the kitti dataset
        ground_truth : True to draw the ground_truth bboxes 
        box_proposals: None if there are no proposals to draw. Otherwise list of [x1,y1,x2,y2] bboxes
            Example: [[0,0,5,5],[10,12,23,25]] or [[3,5,12,16]]
        save: either a name that the image to be saved as or None not to save 
        val_type: None or one of 'easy', 'moderate' and 'hard' only to show objects according to their difficulty to detect
        Example: 
            dataset.show_image(9,ground_truth=True,box_proposals=[[50,190,200,500.],[400,200,200,310]],\
                save="save_name",filter_out_classes=None)
        
        """
        
        img = Image.open(self.image_addresses[ind]).convert("RGB")
        if resize is not None:
            img = img.resize(resize, Image.ANTIALIAS)
        draw = ImageDraw.Draw(img)
        #'Pillow/Tests/fonts/FreeMono.ttf'
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", 15)
        if ground_truth:
            objs = self.get_objects(ind,filter_out_classes)
            points = []
            for ii,obj in enumerate(objs):
                if val_type is None:
                    draw.rectangle(obj.get_2d_bbox(),outline="red",width=2)
                    if print_class_name:
                        #print(obj.class_name)
                        #draw.text([obj.get_2d_bbox()[0]-1,obj.get_2d_bbox()[1]-1],str(ii)+":"+obj.class_name,font=fnt)
                        draw.text([obj.get_2d_bbox()[0]+2,obj.get_2d_bbox()[1]-1],str(ii),font=fnt,fill=(0,153,0,128))
                else:
                    if self.check_object_type(obj,val_type):
                        draw.rectangle(obj.get_2d_bbox(),outline="red",width=2)
                        if print_class_name:
                            print(obj.class_name)
                            #draw.text([obj.get_2d_bbox()[0]+2,obj.get_2d_bbox()[1]-1],str(ii),font=fnt)
                            draw.text([obj.get_2d_bbox()[0]-1,obj.get_2d_bbox()[1]-1],obj.class_name)
        
        if box_proposals is not None:
            for box_proposal in box_proposals:
                draw.rectangle(box_proposal,outline="green",width=2)
        
        if save is not None:
            img.save(save,"PNG")
        plt.imshow(img)
    
    

        
    def iou_func(self,box1,box2):
        x1,y1,x2,y2 = box1
        x3,y3,x4,y4 = box2

        area_box1 = np.abs((x1-x2)*(y1-y2))
        area_box2 = np.abs((x3-x4)*(y3-y4))

        int_box_x1 = min(x2,x4)
        int_box_y1 = min(y2,y4)
        int_box_x2 = max(x1,x3)
        int_box_y2 = max(y1,y3)

        area_intersection = max(0,int_box_x1-int_box_x2)*max(0,int_box_y1-int_box_y2)
        #print(area_box1,area_box2,area_intersection)
        return area_intersection/ (area_box1+area_box2-area_intersection)

    def change_vert(self,rect_xywh):
        """
        Using x,y,w,h values of a rectangular 
        returns x1,y1,x2,y2 points of two opposite vertices
        """
        x1 = rect_xywh[:,0:1]
        y1 = rect_xywh[:,1:2]
        w = rect_xywh[:,2:3]
        h = rect_xywh[:,3:]

        x2 = x1+w
        y2 = y1+h
        return np.hstack((x1,y1,x2,y2)) 
    

    def get_objects(self,ind,filter_out_classes=['Van','Truck','Person_sitting','Tram','Misc','DontCare']):
        """
        ind : the index of the image (or label) that the objects will be read from
        filter_out_classes: To filter out the objects that belong to the given classes
        
        Extracts object ground truth that are read from the label file indicated with the ind (index).
        Creates a kitti object and assigns all the features as object features.
        
        Loops through the lines and splites the string of each line with " " (space). Casts the splitted 
        data into an appropriate type. Sets a kitti object for each line.
        
        Returns the list of Kitti objects.
        """
        try:
            fr_obj_list = self.obj_list[ind]
        except:
            fr_obj_list = []
        objs = []
        for obj in fr_obj_list:
            if filter_out_classes is None or obj.class_name not in filter_out_classes:
                objs.append(obj)
        
        return objs
        

    def get_multimage_objects(self,ind_list):
        multimage_object_list =[]
        for ind in ind_list:
            multimage_object_list.append(self.get_objects(ind))
        return multimage_object_list
        
    def get_id_from_addr(self,addr):
        """
        returns the index part of the label or image address as integer and text
        """
        return int(addr[-10:-4]),addr[-10:-4]
            
    def get_train_val_addr(self,train_perc,shuffle_dataset=False,seed=None):
        """
        train_perc: a float between zero and one indicating the share of training data from all data
        Returns addresses and indices for training (images+labels) and for validation (images+labels)
                [train_image_addr,train_label_addr,val_image_addr,val_label_addr]
        """
        
        mid_ind = int(train_perc*self.len_addr)
        
        if not shuffle_dataset:
                
            train_image_addr = self.image_addresses[0:mid_ind]
            train_label_addr = [i for i in range(mid_ind)]
            
            val_image_addr = self.image_addresses[mid_ind:]
            val_label_addr = [i for i in range(mid_ind,len(self.image_addresses))]
            
        else:
            lbl_indices = [i for i in range(len(self.image_addresses))]
            c = list(zip(self.image_addresses,lbl_indices))
            if seed is not None:
                random.seed(seed)
            random.shuffle(c)
            address,labels = zip(*c)
            
            train_image_addr = address[0:mid_ind]
            train_label_addr = labels[0:mid_ind]
            
            val_image_addr = address[mid_ind:]
            val_label_addr = labels[mid_ind:]
        
        return [train_image_addr,train_label_addr,val_image_addr,val_label_addr]
    def read_img_ind_file(self,path):
        """
        Reads the txt files that contain train-val splits. Txt files are considered to include indices of images on the each line.
        path : Path of the txt file to be read
        
        Returns indices as an integer list
        
        """
        f = open(path,'r')
        lines = f.readlines()
        int_val = [int(line) for line in lines]
        f.close()
        return int_val 
        

class object_kitti_tracking:
    def __init__(self,obj_dict):
        '''
            obj_dict: A dictionary that contains information of the object as shown below
                {'frame':int,'track_id':int,'type':str,'truncated':int,'occluded':int,
                  'alpha':float, 'x1_2d':float,'y1_2d':float,'x2_2d':float,'y2_2d':float,
                  'h_3d':float,'w_3d':float,'l_3d':float,'x_3d':float,'y_3d':float,'z_3d':float,
                  'rotation_y':float}
            Returns an object_kitti_tracking instance
        '''
        self.object_types = ['Car','Van','Truck','Pedestrian','Person_sitting','Cyclist','Tram','Misc','DontCare']
        self.set_all(obj_dict)
    
        
        self.image_shp_x = -1
        self.image_shp_y = -1
    def set_img_shp(self,shp):
        '''
        Sets the shape of the image of this object
            shp: Shape of the image in (h,w,ch) or (h,w) format
       
        '''
        self.image_shp_x = shp[1]
        self.image_shp_y = shp[0]
    
    def set_all(self,obj_dict):
        '''
        Sets labels of the object using the given obj_dict with the labels shown below.
                {'frame':int,'track_id':int,'type':str,'truncated':int,'occluded':int,
                  'alpha':float, 'x1_2d':float,'y1_2d':float,'x2_2d':float,'y2_2d':float,
                  'h_3d':float,'w_3d':float,'l_3d':float,'x_3d':float,'y_3d':float,'z_3d':float,
                  'rotation_y':float}
        
        '''
        self.frame = obj_dict['frame']
        self.track_id = obj_dict['track_id']
        self.text_frame = '{:06d}'.format(self.frame)
        if obj_dict['type'] == 'Person':
            self.class_name = 'Pedestrian'
        else:            
            self.class_name = obj_dict['type']
        
        if self.class_name in self.object_types:
            self.class_id = self.object_types.index(self.class_name)
        else:
            self.class_name = 'Misc'
            self.class_id = self.object_types.index(self.class_name)
        self.truncated = obj_dict['truncated']
        self.occluded = obj_dict['occluded']
        self.alpha = obj_dict['alpha']
        
        self.x1 = obj_dict['x1_2d']
        self.y1 = obj_dict['y1_2d']
        self.x2 = obj_dict['x2_2d']
        self.y2 = obj_dict['y2_2d']
        
        self.h = obj_dict['h_3d']
        self.w = obj_dict['w_3d']
        self.l = obj_dict['l_3d']
        
        self.x = obj_dict['x_3d']
        self.y = obj_dict['y_3d']
        self.z = obj_dict['z_3d']
        self.rotation_y = obj_dict['rotation_y']
        
        self.check_val_type()

        
    def get_2d_bbox(self):
        """
        returns bounding boxes in the format of [x1,y1,x2,y2]
        """
        return [self.x1,self.y1,self.x2,self.y2]
    
    def get_prop_box(self):
        """
        returns the proportional position of bounding box, each of which is a value between 0 and 1.
        """
        return [self.x1/self.image_shp_x,self.y1/self.image_shp_y,self.x2/self.image_shp_x,self.y2/self.image_shp_y]
    
    def get_2d_hw(self):
        """
        returns bounding boxes in the format of [x1,y1,w,h]
        """
        return [self.x1,self.y1,self.x2-self.x1,self.y2-self.y1]
    
    def get_2d_center(self):
        """
        returns bounding boxes in the format of [x_center,y_center,w,h]
        """
        return [(self.x1+self.x2)/2.0,(self.y1+self.y2)/2.,self.x2-self.x1,self.y2-self.y1]
    
    def check_val_type(self):
        if np.abs(self.y1-self.y2)>=40 and self.truncated<=0.15 and (self.occluded in [0]):
            self.val_type = 'easy'
            self.val_type_id = 0
        elif np.abs(self.y1-self.y2)>=25 and self.truncated<=0.3 and (self.occluded in [0,1]):
            self.val_type = 'moderate'
            self.val_type_id = 1
        elif np.abs(self.y1-self.y2)>=25 and self.truncated<=0.5 and (self.occluded in [0,1,2]):
            self.val_type = 'hard'
            self.val_type_id = 2
        else:
            self.val_type = 'unknown'
            self.val_type_id = 3     
    def print_summary(self):
        print("Image idx: {}, Image name: {}, Track ID:{}, Class name: {}, Class idx: {}, \
              Truncation: {}, Occlusion: {}, Alpha: {}, x1: {}, y1: {}, x2: {},\
              y2: {}, h: {}, w: {}, l: {}, x: {}, y: {}, z: {}, Rotation y: {},\
              Difficulty type: {}, Difficulty type idx: {}".format(self.frame,self.text_frame,self.track_id,\
                                                                  self.class_name,self.class_id,\
                                                                  self.truncated,self.occluded,\
                                                                  self.alpha,self.x1,self.y1,self.x2,\
                                                                  self.y2,self.h,self.w,self.l,self.x,\
                                                                  self.y,self.z,self.rotation_y,\
                                                                  self.val_type,self.val_type_id))
        
        
def plt_im(im):
    plt.imshow(im.astype(np.uint8))
        
        
def get_dataset(video_num=0,main_path=None):
    '''
    video_num : ID of the drive in KITTI tracking dataset
    main_path : Path to the folder that contains data_tracking_label_2 and data_tracking_image_2 folders, inside which is labels and images in the appropriate format.
    
    Returns the dataset instance for tracking
    '''
    if main_path is None:
        main_path = 'tracking_dataset/KITTI/'
    label_path = main_path + 'data_tracking_label_2/training/label_02/{:04d}.txt'.format(video_num)
    image_path = main_path + 'data_tracking_image_2/training/image_02/{:04d}/*.png'.format(video_num)
    
    dataset = obj_detection_dataset(image_path=image_path,label_path=label_path,test_data=False)
    return dataset

def get_frame_data(frame_id,dataset,filter_out_classes=[]):
    '''
    frame_id : Frame index of the desired ground-truth objects
    dataset  : Dataset instance for KITTI tracking
    filter_out_classes: A list of class names that won't be included in the object list
    
    Returns numpy arrays of ground-truth bounding boxes (n,4), class names (n), tracking IDs (n)
    
    '''
    objs = dataset.get_objects(ind=frame_id,filter_out_classes=filter_out_classes)
    bboxes = []
    class_names = []
    track_ids = []
    for obj in objs:
        bboxes.append(obj.get_2d_bbox())
        class_names.append(obj.class_name)
        track_ids.append(obj.track_id)
        
    return np.array(bboxes),np.array(class_names),np.array(track_ids)

def get_all_frame_objs(dataset,filter_out_classes=[]):
    '''
    To get ground-truth objects of all frames of a drive in KITTI tracking dataset
    
    dataset : Dataset instance for KITTI tracking dataset
    filter_out_classes : Class names that won't be included in the ground-truth object list
    
    Returns 
        bbox_fr: List of ground-truth bounding boxes of every frame [np_array_of_frame0(n,4), np_array_of_frame1 (m,4), ...]
        clss_fr: List of ground-truth class names of every frame [np_array_of_frame0(n), np_array_of_frame1 (m), ...]
        inds_fr: List of ground-truth tracking IDs of every frame [np_array_of_frame0(n), np_array_of_frame1 (m), ...]
    '''
    bbox_fr = []
    clss_fr = []
    inds_fr = []
    for i in range(dataset.len_addr):
        box,cls,ind = get_frame_data(frame_id=i,dataset=dataset,filter_out_classes=filter_out_classes)
        bbox_fr.append(box)
        clss_fr.append(cls)
        inds_fr.append(ind)
    return bbox_fr,clss_fr,inds_fr


### Using extracted features, generate sequences of objects

def add_to_sequence(ind_cond,time_step,seqs,seqs_available,new_seqs_available,verbose=1):
    '''
    Method to add detected objects according to the matching condition with the previously detected objects
    
    ind_cond : A numpy array of (n,2) showing indices of n object matches. Col0 shows ind of matched obj in the current frame, Col1 shows in of matched obj in the prev frame.
    time_step: Current time step. (Integer)
    seqs     : List of sequences up to now. Each sequence is a list of lists with two integers. The first integer shows the time step and the second integer shows the object. [[[0,1],[1,2],[2,1],[3,2]],[[1,3],[2,3],[3,0]],[...],[...]] -> Four sequences. The first sequence relates obj1 of frame 0, obj2 of frame 1, obj1 of frame 2 and obj2 of frame 3.
    seqs_available : List of integers showing indices of sequences that are available for appending, which means that contains objects from the previous frame (not dead-end).
    new_seqs_available: List to append indices of sequences that objects from the current frame appended. 
    
    '''
    seqs_to_remove=[]
    new_seq_for_ind = []
    new_sequences = []
    # Add object matches into the sequences
    #print(seqs_available)
    for ind_seq in seqs_available:
        for obj_match in ind_cond:
            ## If the matched obj is a member of any of the available sequences
            #print(obj_match)
            if seqs[ind_seq][-1] == [time_step-1,obj_match[1]]:
                #print(obj_match)
                seqs_to_remove.append(seqs[ind_seq])
                new_seq = seqs[ind_seq]
                new_seq.append([time_step,obj_match[0]])
                new_sequences.append(new_seq)
                new_seq_for_ind.append(new_seq)
                #new_seqs_available.append(len(seqs)-1)
                #seqs[ind_seq].append([time_step,obj_match[0]])
                #new_seqs_available.append(ind_seq)
    
    for seq_to_remove in seqs_to_remove:
        seqs.remove(seq_to_remove)
    seqs += new_sequences
    
    
    for n_seq in new_sequences:
        new_seqs_available.append(seqs.index(n_seq))
    if verbose>=2:
        print("Seq to remove:",seqs_to_remove)
        print("Sequences:", seqs)
        print("Availability ind:",new_seqs_available)
        print()

def trim_seq(sequence,min_obj_num=2):
    '''
    To trim the sequences that have smaller number of objects than min_obj_num (len(seq)<min_obj_num)
    
    min_obj_num : Minumum number of objects in a tracked sequence
    
    Returns trimmed sequences
    Example : new_sequence = trim_seq(sequence=sequences,min_obj_num=2)
    '''
    ## Eliminate sequences that have only one object assignment
    new_sequences = []
    for i,seq in enumerate(sequence):
        if len(seq)>=min_obj_num:
            new_sequences.append(seq)
    return new_sequences

def print_sequence(sequence):
    '''
    To print number of detections in a sequence and the sequence itself
    The sequence contains pairs of (frame_id,detected_obj_id_in_that_frame)
    '''
    for i,seq in enumerate(sequence):
        print("Sequence {} has {} number of detections".format(i,len(seq)))
        print("Sequence: ",seq)
        print()
        
def print_and_save_images(path,dataset,results,sequence,gt_seqs):
    '''
    path = "tracking_dataset/KITTI/tracking_tfrecords/seq_drive-{}".format(video_num)
    print_and_save_images(path,dataset,results,new_sequence,seq_gt)
    
    '''
    ind_features=3
    ind_boxes = 0
    ind_proposals = 5
    ind_scores = 1
    ind_classes = 2
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)
    #dataset.get_image_2()
    for i,(seq,gt_seq) in enumerate(zip(sequence,gt_seqs)):
        print("Sequence {} has {} number of detections".format(i,len(seq)))
        for i_obj,(s,gt) in enumerate(zip(seq,gt_seq)):
            print("{}: {}".format(s,gt))
            im = dataset.get_image_2(s[0])
            det_box = [results[s[0]][ind_boxes][s[1]],gt[0]]
            det_cls = [results[s[0]][ind_classes][s[1]],gt[1]]
            det_scr = [results[s[0]][ind_scores][s[1]],gt[2]]
            show_image(im[0].astype(np.uint8),box_proposals=det_box,class_names=det_cls,
                       class_scores=det_scr,save=path+"/"+"{}-{}.png".format(i,i_obj))
        print()
    
def print_sequence_with_gt(sequence,gt_seqs):
    '''
    To print number of detections in a sequence and the sequence itself
    The sequence contains pairs of (frame_id,detected_obj_id_in_that_frame)
    '''
    for i,(seq,gt_seq) in enumerate(zip(sequence,gt_seqs)):
        print("Sequence {} has {} number of detections".format(i,len(seq)))
        for s,gt in zip(seq,gt_seq):
            print("{}: {}".format(s,gt))
        print()
        
def gen_bbox_feat(fr_ind,obj_bbox,dataset,verbose=0):
    '''
    From the bounding boxes generates features
        bbox [x_c,y_c,w,h]
        image [im_h,im_w]
        bbox features  [x_c/im_w, y_c/im_h, w/im_w, h/im_h]
        
    '''
    im,(im_h,im_w,im_ch) = dataset.get_image_2(fr_ind)
    
    obj_x_c, obj_y_c, obj_w, obj_h = np_corner2center(obj_bbox)
    scl_x_c = obj_x_c / im_w
    scl_y_c = obj_y_c / im_h
    scl_w = obj_w / im_w
    scl_h = obj_h / im_h
    if verbose>=2:
        print("Obj dims: x center:{}, y center:{}, w:{}, h:{}".format(obj_x_c,obj_y_c, obj_w, obj_h))
        print("Scaled obj dims: x center:{}, y center:{}, w:{}, h:{}".format(scl_x_c,scl_y_c, scl_w, scl_h))
    return np.array([scl_x_c, scl_y_c, scl_w, scl_h])        

def np_corner2center(bbox):
    '''
    bbox in [x1,y1,x2,y2] format
    
    returns the bbox in [x_center,y_center, w,h] format
    
    '''
    return [(bbox[0]+bbox[2])/2.,(bbox[1]+bbox[3])/2.,bbox[2]-bbox[0],bbox[3]-bbox[1]]

def gen_lstm_data(features,detections,gt_bboxes,gt_labels):
    '''
    From the features of sequences generate a dataset ready to feed lstm network training.
    
    features  : [n_seq,tau,len_feature_vec]
    detections: [n_seq,4]
    gt_bboxes : [n_seq,4]
    gt_labels : [n_seq,1]
    
    returns a tuple ([features,detections],np.concatenate((gt_bboxes,detections,gt_labels),axis=1))
    
    '''
    data = ([features,detections,gt_labels],
             np.concatenate((gt_bboxes,detections,gt_labels),axis=1))
    return data

def gen_features(main_path,dataset_path,video_nums,proposal_type='non-gt',pred_thr=0.5,
                 iou_thr1=0.5,iou_thr2=0.15,iou_obj_gt=0.4,cls_names=['Bg','Car','Pedestrian','Cyclist'],
                 tau=5,ext_features=True,concat_feat=False,save_lstm_data=True,return_arr=True,verbose=1):
    '''
    main_path   : Path that shows the folders that contain raw feature maps previously saved. 
        Ex: main_path = 'features/', inside this folder feature folders should be formatted as following:
        features/video_feature_extraction/drive_{drive_num:04d}/drive_{drive_num:04d}_preds_video_feature_{pred_thr}_{proposal_type}/features.npy'
        proposal_type is either 'gt' or 'non-gt'
    dataset_path: Path to the folder that contains data_tracking_label_2 and data_tracking_image_2 folders, inside which is labels and images in the appropriate format.
        
    video_nums   : List of indices that enumerate dataset from different drives in KITTI.
    proposal_type: 'gt' or 'non-gt'. 'gt' means that noise-added ground-truth bounding boxes are provided to the detector as proposals. 'non-gt' means that proposals are from RPN.
    pred_thr     : Prediction score threshold of the object detector. This is only necessary since the folder is saved with this parameter. This means that predicted objects with a lower score are ignored.
    iou_thr1     : IoU threshold to assign multiple objects detected in frame_t+1 to one object in frame_t, which will generate multiple sequences of the same object that is detected with multiple bounding boxes in the following frame. 
    iou_thr2     : Objects of frame_t+1 that have an IoU threshold between iou_thr1 and iou_thr2 will be assigned to the object of frame_t if the object of frame_t was not assigned to another object. In this case the object of frame_t will generate only 1 sequence, which will continue with the maximum overlapping object of frame_t+1.
    iou_obj_gt   : IoU threshold to assign detections to the ground-truths. Objects that have an IoU value with a gt box above this threshold will be assigned as objects
    cls_names    : Class names, only sequences of which will be included in the sequence list. In addition to the classes in KITTI, 'Bg' is used to describe background.
    tau          : Number of time steps that will be included in the features arranged and sampled for LSTM training. feature tensor will have a shape of [num_sequences,tau,len_feature_vec]
    ext_features : True to extend features. Normally only features coming from detector will be used. If True, these features will be extended with the prediction score and scaled predicted bounding box in the center format.
    concat_feat  : If there are data from more than one drive, this flag can be used to concat features as coming from one video. If True, all the features and detections will be concatenated. Only num_sequences will increase, other dimensions will remain same. 
    save_lstm_data: Saves the generated lstm data into the main_path with the name of videos if True. concat_feat should be True as well.
    return_arr   : To return recorded arrays for all videos
    verbose      : To print our summary (0: No prints, 1: Only important, 2: All possible)
    '''
    
    ## To have more sequences change how num_sample defined below (in line 936)
    
    ## Create tensorflow graph for IoU calculation
    boxes1 = tf.placeholder(shape=(None,4),dtype=tf.float32,name='box1')
    boxes2 = tf.placeholder(shape=(None,4),dtype=tf.float32,name='box2')
    # iou,boxAArea, boxBArea, interArea
    iou,areaA,areaB,intArea = tf_iou(boxes1,boxes2)
    
    returns_of_videos = []
    ## To keep numbers of sequences and objects that belong to each of the classes
    num_sequences = []
    num_instances = []
    final_seq_counts = []
    ## Iterate through features of the desired drives
    for i_vid, video_num in enumerate(video_nums):
        if verbose>=1:
            print("\nStarting generating features of drive {:04d}".format(video_num))
        
        ### Load features
        feature_path = main_path+'video_feature_extraction/drive_{:04d}/drive_{:04d}_preds_video_feature_{}_{}/features.npy'.format(video_num,video_num,pred_thr,proposal_type)
        np_features = np.load(feature_path)
        
        num_timesteps = len(np_features)
        
        ### Assign detections to each other in successive frames
        sequences = [] # List of sequences -> [seq1, seq2, ...] -> seq1: [[fr_id,obj_id],[fr_id+1,obj_id], ...],...
        # Indices showing the related information in the np_features list
        ind_boxes = 0 
        ind_scores = 1
        ind_classes = 2
        ind_features=3
        ind_proposals = 5
        ## TODO: Instead of IoU, I can check Intersection/Gt area. This might make more sense.
        ### Calculate IoU between objects in successive frames
        with tf.Session() as sess:
            for n_time in range(0,num_timesteps):
                # Add objects in the first frame to the sequences
                if n_time == 0:
                    n_b = len(np_features[n_time][ind_boxes])
                    # available_sequences: Shows the index of the sequence in the sequences list that 
                    # new detections can be added. The condition to be added is to have a detection with
                    # a related IoU in the last frame. There shouldn't be any skips in frames.
                    available_sequences=[] 
                    for i in range(n_b):
                        sequences.append([[n_time,i]])
                        available_sequences.append(i)
                else:
                    new_available_sequences=[]
                    # IoU  btw. the current frame and the prev. frame. Rows: Obj in curr. frame, Cols: Obj in prev. frame
                    iou_val = sess.run(iou,feed_dict={boxes1:np_features[n_time][ind_boxes],
                                                      boxes2:np_features[n_time-1][ind_boxes]})
                    #iou_val = res_iou[0] 

                    cond1 = iou_val>=iou_thr1
                    # Cond2 is True for all the entries that have no Trues in their rows in cond1 and satisfy the cond2, which is IoU>=iou_thr2 and IoU<iou_thr1
                    cond2 = np.logical_and(np.logical_and(iou_val>=iou_thr2,iou_val<iou_thr1), np.logical_not(np.any(iou_val>=iou_thr1,axis=1,keepdims=True)))
                    # Cond3 is True for all the entries that have no Trues in their rows in cond1 and cond2.
                    cond3 = np.logical_not(np.any(np.logical_or(cond1,cond2),axis=1,keepdims=True))


                    # If there are objects in the current frame that have IoU>=iou_thr1 with the objects of prev. frame
                    if np.any(cond1):
                        ind_cond1 = np.argwhere(cond1) # [[Ind_obj_curr_frame,ind_obj_prev_frame], ...]
                        if np.any(cond2):
                            ind_cond2 = np.argwhere(cond2)
                            ind_cond = np.concatenate((ind_cond1,ind_cond2),axis=0)
                        else:
                            ind_cond = ind_cond1
                    else:
                        if np.any(cond2):
                            ind_cond = np.argwhere(cond2)
                        else:
                            ind_cond = None


                    if ind_cond is not None:
                        if verbose>=2:
                            print("Frame number:",n_time)
                            print("IoU values (Rows: Frame{}, Cols: Frame{})".format(n_time,n_time-1))
                            print(np.around(iou_val,decimals=2))
                            print("Available seqs:",available_sequences)
                            print("New available seqs:",new_available_sequences)
                        # Add object matches into the sequences
                        add_to_sequence(ind_cond,n_time,sequences,available_sequences,new_available_sequences)
                    
                    # Objects of current frame that is not part of any sequences
                    if np.any(cond3):
                        ind_cond3 = np.argwhere(cond3)
                        for i_cond3 in ind_cond3:
                            sequences.append([[n_time,i_cond3[0]]])
                            new_available_sequences.append(len(sequences)-1)

                    available_sequences = new_available_sequences
        
        ### Eliminate sequences that have only one object
        trimmed_sequence = trim_seq(sequence=sequences,min_obj_num=2)
        
        ### Assign objects in the sequence to the ground-truth bounding boxes
        
        # Generate dataset related instances
        dataset = get_dataset(video_num=video_num,main_path= dataset_path)
        # Gt information of all frames that is extracted using the dataset instance
        gt_boxes,gt_classes,gt_indices = get_all_frame_objs(dataset=dataset,filter_out_classes=[])
        
        trimmed_seq_gt = [] # Ground-truths of each sequence will be stored in this list with the same format
        # [gt_seq1, gt_seq2, ...] -> gt_seq1: [gt_obj1, gt_obj2, ...] -> gt_obj1: [x1,y1,x2,y2,cls_name,trck_id]
        with tf.Session() as sess:
            for seq in trimmed_sequence:
                trimmed_seq_gt.append([]) # Generate new sequence for every sequence in trimmed_sequence
                for time_step,obj_id in seq:
                    bbox = np_features[time_step][ind_boxes][obj_id:obj_id+1]
                    gt_b = gt_boxes[time_step]
                    gt_c = gt_classes[time_step]
                    gt_i = gt_indices[time_step]
                    res_iou,res_A,res_B,res_int = sess.run([iou,areaA,areaB,intArea],feed_dict={boxes1:bbox,boxes2:gt_b})
                    
                    ## This is the IoU value to match obj with the gts
                    #iou_val = res_iou[0] # IoU of the object with ground-truths
                    
                    ## This is the intersect/gt_area value to match obj with the gts
                    ## I think intersect/gt_area is a better metric
                    res_int = np.reshape(res_int,(-1))
                    res_B = np.reshape(res_B,(-1))
                    if verbose>=3:
                        print("Sequence:{}, Obj ID:{}".format(time_step,obj_id))
                        print("     {}".format(np.around(res_int/res_B,2)))
                        
                    
                    iou_val = res_int/res_B
                    
                    ind_max = np.argmax(iou_val)
                    if iou_val[ind_max]>=iou_obj_gt:
                        trimmed_seq_gt[-1].append([gt_b[ind_max],gt_c[ind_max],gt_i[ind_max]])
                    else:
                        ## This is just to avoid confusion with Dontcare since the dontcare regions are not objects specific and large. Therefore, the iou criterion can't be met.
                        if gt_b[ind_max] == 'Dontcare' and iou_val[ind_max]>iou_obj_gt/2.:
                            trimmed_seq_gt[-1].append([[-1,-1,-1,-1],'Dontcare',-1])
                        else:
                            trimmed_seq_gt[-1].append([[-1,-1,-1,-1],'Bg',-1])
        
        
        ### Split the sequences according to the assigned ground-truth classes
        ### Means that sequences that have objects with different gt tracking ID will be 
        ### split into difference sequences
        
        ## TODO: Here instead of tracking ID I use class name. It should be tracking ID to split
        split_seq = []
        split_gt = []
        for i_seq,(seq,gt_seq) in enumerate(zip(trimmed_sequence,trimmed_seq_gt)):
            split_seq.append([])
            split_gt.append([])  
            len_seq = len(seq)
            if len_seq>=2:
                for i_pair,(s,gt_s) in enumerate(zip(seq,gt_seq)):
                    if i_pair == 0:
                        last_cls = gt_s[-2]
                        last_ind = gt_s[-1]
                        split_seq[-1].append(s)
                        split_gt[-1].append(gt_s)
                    else:
                        ## Check if classes and tracking ids are matched
                        ## Checking both classes and tracking ids is important due to Dontcare and Bg classes, both of which have -1 as tracking ID
                        if (last_cls == gt_s[-2]) and (last_ind == gt_s[-1]) :
                            split_seq[-1].append(s)
                            split_gt[-1].append(gt_s)
                        else:
                            last_cls = gt_s[-2]
                            last_ind = gt_s[-1]
                            split_seq.append([])
                            split_gt.append([])
                            split_seq[-1].append(s)
                            split_gt[-1].append(gt_s)
        
        ### Trim the split sequences to eliminate sequences with only 1 object
        trimmed_split_seq = trim_seq(sequence=split_seq,min_obj_num=2)
        trimmed_split_seq_gt = trim_seq(sequence=split_gt,min_obj_num=2)
        

        
        ### Select sequences with certain classes
        num_seq = [0 for i in range(len(cls_names))] # Number of sequences that belong to each cls
        num_instance = [0 for i in range(len(cls_names))] # Number of instances that belong to each cls
        filtered_seq = []
        filtered_gt = []
        for s,gt in zip(trimmed_split_seq,trimmed_split_seq_gt):
            if gt[0][1] in cls_names:
                ind_cls = cls_names.index(gt[0][1])
                num_seq[ind_cls]+=1
                num_instance[ind_cls]+=len(s)
                filtered_seq.append(s)
                n_gt = []
                for g in gt:
                    n_gt.append( g + [ind_cls])
                filtered_gt.append(n_gt)
            else:
                pass
        if verbose>=1:
            for i_cls,cls in enumerate(cls_names):
                print("{} class has {} objects in {} sequences.".format(cls,num_instance[i_cls],num_seq[i_cls]))
            print('In total, {} objects in {} sequences.'.format(np.sum(num_instance),np.sum(num_seq)))
        num_sequences.append(num_seq)
        num_instances.append(num_instance)
            
        
        
        ### Features and related information called according to the frame and obj information in seq list
        filtered_features=[] # Feature vectors
        filtered_scores = [] # Scores
        filtered_classes=[] # Predicted classes
        filtered_proposals=[] # Predicted bounding boxes (from object detector)

        for seq in filtered_seq:
            filtered_features.append([])
            filtered_scores.append([])
            filtered_classes.append([])
            filtered_proposals.append([])
            for s in seq:
                filtered_features[-1].append(np_features[s[0]][ind_features][s[1]])
                filtered_scores[-1].append(np_features[s[0]][ind_scores][s[1]])
                filtered_classes[-1].append(np_features[s[0]][ind_classes][s[1]])
                filtered_proposals[-1].append(np_features[s[0]][ind_proposals][s[1]])
                
                
        
        
        ### Collect features and gt information for RNN training
        ### Padding will be done here with zeros
        ### Number of features will be aligned according to tau with either padding or sampling from the sequence
        features = []
        detections = []
        gt_bboxes = []
        gt_labels = []
        frames= []

        # last tau frame to feed into LSTMs. If a sequence doesn't contain tau number of detections, the initial features will be padded with zeros.
        if ext_features:
            len_feature_vec = len(filtered_features[0][0]) + 1 + 4 # 4096: feature vector from detector, 1: predicted score, 4: predicted bbox
        else:
            len_feature_vec = len(filtered_features[0][0])
        
        for i_seq, (seq,seq_gt,seq_feat,seq_scr,seq_cls,seq_props) in enumerate(zip(filtered_seq,filtered_gt,
                                                                                filtered_features,filtered_scores,
                                                                                filtered_classes,filtered_proposals)):

            n_frame = len(seq) # number of objects (frames) in the sequence 

            #### Generate features



            # If the sequence longer than the desired tau, then sample tau number of objects from the sequence
            # Number of the resulting sequences is then int(n_frame/tau)
            if n_frame>tau:
                st_indices = [i for i in range(n_frame-tau+1)]
                #st_indices = np.random.choice(st_indices,size=(int(n_frame/tau))) # from which index the new sequence start
                num_sample = np.max([1,int((n_frame-tau))])
                st_indices = np.random.choice(st_indices,replace=False,size=(num_sample)) # from which index the new sequence start
                n_iterate = tau # tau times iterated to add feature vectors into the feature list
            else:
                st_indices = [0] # If number of objects in the sequence is less than tau, the feature matrix of the sequence will be generated using the first entry in the sequence
                n_iterate = n_frame

            for st_ind in st_indices:
                features.append([])
                final_ind = st_ind+n_iterate-1 # Index pointing the last object of the sequence that has tau frames (objects). This shows the index of last object of the new sequence, which is a subset of the considered and split sequence.
                detections.append(seq_props[final_ind]) # For each list in features there is one proposal (detection), which is the detection on the last frame
                gt_bboxes.append(seq_gt[final_ind][0]) # For each list (sequence-> time_step*len_features) in features, there is only one gt bbox, which is the gt of the detection in the last frame
                gt_labels.append(seq_gt[final_ind][3]) # Same with gt bbox, only one label for each sequence, which shows the class of the gt bbox on the last frame
                frames.append([video_num,seq[final_ind][0]])
                list_local_feat = []
                # Pad the feature vector with zeros to match the shape of features (time_step,len_features)
                for i_pad in range(tau-n_frame):
                    list_local_feat.append(list(np.zeros(shape=(len_feature_vec))))
                
                # Iterate over the real features to fill in the feature matrix with the extended or real features (time_step, len_features)
                # For each entry in the detections, gt_bboxes,gt_labels lists there is one matrix of features 
                # This for loop is for each row in the feature matrix
                for i_iterate in range(n_iterate):
                    ind_obj = i_iterate+st_ind ## Start iterating objects in the sequence from this index
                    fr_ind,obj_ind  = seq[ind_obj] # Object pairs in the sequence [frame_ind, object_ind]
                    obj_bbox = seq_props[ind_obj] ## detected bbox
                    # bbox_feat is the scaled bbox in center format
                    bbox_feat = gen_bbox_feat(fr_ind,obj_bbox,dataset,verbose=verbose)
                    if ext_features:
                        feat = np.concatenate((seq_feat[ind_obj], [seq_scr[ind_obj]], bbox_feat ))
                    else:
                        feat = seq[ind_obj]
                    list_local_feat.append(list(feat))

                features[-1] = np.reshape(list_local_feat,newshape=(tau,len_feature_vec))

                #features[-1] = np.reshape(features[-1],newshape=(tau,len_feature_vec))




        features = np.reshape(features,newshape=(-1,tau,len_feature_vec))
        detections = np.array(detections)
        gt_bboxes = np.array(gt_bboxes)
        gt_labels = np.reshape(gt_labels,newshape=(-1,1))
        frames = np.reshape(frames,newshape=(-1,2))
        
        
        
        ### Concatenate features from all videos
        if concat_feat:
            if i_vid == 0:
                concat_features = features
                concat_detections = detections
                concat_gt_bboxes = gt_bboxes
                concat_gt_labels = gt_labels
                concat_frames = frames
            else:
                concat_features = np.concatenate((concat_features,features),axis=0)
                concat_detections = np.concatenate((concat_detections,detections),axis=0)
                concat_gt_bboxes = np.concatenate((concat_gt_bboxes,gt_bboxes),axis=0)
                concat_gt_labels = np.concatenate((concat_gt_labels,gt_labels),axis=0)
                concat_frames = np.concatenate((concat_frames,frames),axis=0)
        
        
        ### Return 
        ### - sequences that shows the frame-object pairs in a sequence list
        ### - trimmed_sequence that shows only the sequences with more than one object
        ### - trimmed_seq_gt that shows ground-truths of trimmed_sequences
        ### - split_seq that shows sequences split into sub-sequences if objects 
        ### inside a sequence are assigned to gts with different trck IDs
        ### - split_gt that shows ground-truths of split_seq
        ### - trimmed_split_seq that shows trimmed list of split seq
        ### - trimmed_split_seq_gt that shows gts of trimmed_split_seq
        ### - filtered_seq that shows trimmed_split_seq with only sequences of classes in the class_names
        ### - filtered_seq_gt that shows gts of filtered_seq
        ### - filtered_features that shows features of obj in sequences with the same order and in the same format
        ### - filtered_scores that shows scores (probabilities from the detector) of objs in sequences with the same order and in the same format
        ### - filtered_classes that shows classes (predicted classes) of objs in sequences with the same order and in the same format
        ### - filtered_proposals that shows proposals (object detector detections) of objs in sequences with the same order and in the same format
        ### - features that shows list of feature matrices of each sequence. The shape is (n_seq,tau,len_feature_vector)
        ### - detections that shows object bbox predictions from the object detector. Each detection in this list belongs to a feature matrix in features such that the detection shows the predicted object on the last frame of the sequence.
        ### - gt_bboxes that shows ground-truth bbox of every detection in detections list.
        ### - gt_labels that shows ground-truth labels of every detection in detections list.
        ### - frames that shows video-id and frame-id of the objects in the sequences. Frame-ids only shown for the last frame of the sequences.
        ### - concat_features that shows concatenated features from all videos 
        ### - concat_detections that shows concatenated detections from all videos 
        ### - concat_gt_bboxes that shows concatenated gt_bboxes from all videos 
        ### - concat_gt_labels that shows concatenated gt_labels from all videos 
        if return_arr:
            dict_video = {'sequence':sequences,
                          'trimmed_sequence':trimmed_sequence,
                          'trimmed_seq_gt':trimmed_seq_gt,
                          'dataset':dataset,
                          'split_seq':split_seq,
                          'split_gt':split_gt,
                          'trimmed_split_seq':trimmed_split_seq,
                          'trimmed_split_seq_gt':trimmed_split_seq_gt,
                          'filtered_seq':filtered_seq,
                          'filtered_gt':filtered_gt,
                          'filtered_features':filtered_features,
                          'filtered_scores':filtered_scores,
                          'filtered_classes':filtered_classes,
                          'filtered_proposals':filtered_proposals,
                          'features':features,
                          'detections':detections,
                          'gt_bboxes':gt_bboxes,
                          'gt_labels':gt_labels,
                          'frames':frames,
                          'concat_features':concat_features,
                          'concat_detections':concat_detections,
                          'concat_gt_bboxes':concat_gt_bboxes,
                          'concat_gt_labels':concat_gt_labels,
                          'concat_frames':concat_frames,
                          'num_sequences':num_sequences,
                          'num_instances':num_instances,
                          'np_features':np_features    }
            
            returns_of_videos.append(dict_video)
            del dict_video
        
        if verbose>=1:
            print("Num sequences: {}".format(len(filtered_features)))
            final_seq_count = np.zeros((len(cls_names)))
            
            for gt_l in np.reshape(gt_labels,(-1)):
                final_seq_count[gt_l]+=1
            for c,cou in zip(cls_names,final_seq_count):
                print("{} class # of seq:{}".format(c,cou))
            final_seq_counts.append(final_seq_count)
            '''
            returns_of_videos.append([sequences, trimmed_sequence, trimmed_seq_gt, dataset,
                                     split_seq,split_gt,trimmed_split_seq,trimmed_split_seq_gt,
                                     filtered_seq,filtered_gt,
                                      filtered_features,filtered_scores,filtered_classes,filtered_proposals,
                                     features,detections,gt_bboxes,gt_labels,frames,
                                     concat_features,concat_detections,concat_gt_bboxes,concat_gt_labels,
                                     concat_frames,num_sequences,num_instances])
            '''
    ## Generate training data
    ## This will be only generated if there is only one video or all the video features are concatenated
    ## Otherwise returned features should be individually converted into the related tuple
    if concat_feat:
        ### Generate LSTM data
        lstm_data = gen_lstm_data(concat_features,concat_detections,concat_gt_bboxes,concat_gt_labels)
        returns_of_videos.append({'lstm_data':lstm_data})
        if save_lstm_data:
            folder = main_path+'lstm_data'
            if os.path.isdir(folder):
                pass
            else:
                os.makedirs(folder)
            name_ext = '/lstm_data_drives'
            name_ext_frames = '/frame_data_drives'
            for i_vid in video_nums:
                name_ext+='_{:04d}'.format(i_vid)
                name_ext_frames+='_{:04d}'.format(i_vid)
            
            np.save(folder+name_ext,lstm_data)
            np.save(folder+name_ext_frames,concat_frames)
            print("\nData saved as {}".format(folder+name_ext+".npy"))
            print("\nFrames saved as {}".format(folder+name_ext_frames+".npy"))

    if verbose>=1:
        print(cls_names)
        print(np.sum(final_seq_counts,axis=0))
        
        
    if return_arr:
        return returns_of_videos
    else:
        del returns_of_videos
        
def print_box(time_ind,results,iou_values):
    '''
    To print IoU values of the detections with the boxes from the next step 
    Example:
    ## Print IoU values Rows: Curr. frame, Cols: Next frame
    iou_values = []
    with tf.Session() as sess:
        for n_time in range(batch_num-1):
            res_iou = sess.run(iou,feed_dict={boxes1:results[n_time][ind_boxes],boxes2:results[n_time+1][ind_boxes]})
            iou_values.append(res_iou)
            print_box(n_time,results,iou_values)

    '''
    ind_features=3
    ind_boxes = 0
    ind_proposals = 5
    ind_scores = 1
    ind_classes = 2
    iou_ind = 0
    print("Ind:{}".format(time_ind))
    box = results[time_ind][ind_boxes]
    for i,b in enumerate(box):
        scr = results[time_ind][ind_scores][i]
        cls = results[time_ind][ind_classes][i]
        iou_val = iou_values[time_ind][iou_ind][i]
        iou_str = 'IoU: '
        for i_val,val in enumerate(iou_val):
            iou_str += 'B{}:{:.2f} '.format(i_val,val)
        print("Cls:{}, Scr:{:.2f}, Box:{}".format(cls,scr,b))
        print("\t{}".format(iou_str))
    print("\n")         

        
