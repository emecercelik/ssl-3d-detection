
from __future__ import print_function

import numpy as np
import cv2
import os
import sys
from lstm_seq_data import get_dataset
import IPython

class Object3d(object):
    ''' 3d object label '''

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
              (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
              (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
              (self.t[0], self.t[1], self.t[2], self.ry))

class Object3d_video(object):
    ''' 3d object label for tracking dataset '''

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[3:] = [float(x) for x in data[3:]]
        #frame and track id
        self.frame = int(data[0])
        self.track_id = int(data[1])
        
        # extract label, truncation, occlusion
        self.type = data[2]  # 'Car', 'Pedestrian', ...
        self.truncation = data[3]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[4])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[5]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[6]  # left
        self.ymin = data[7]  # top
        self.xmax = data[8]  # right
        self.ymax = data[9]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[10]  # box height
        self.w = data[11]  # box width
        self.l = data[12]  # box length (in meters)
        self.t = (data[13], data[14], data[15])  # location (x,y,z) in camera coord.
        self.ry = data[16]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.check_object_type() # The difficulty level
    def gen_label_line(self):
        self.label_line = '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(self.frame,\
                          self.track_id,self.type,self.truncation,self.occlusion,self.alpha,\
                          self.xmin,self.ymin,self.xmax,self.ymax,self.h,self.w,self.l,self.t[0],self.t[1],self.t[2],self.ry)
        return self.label_line
    def gen_label_line_det(self):
        self.label_line_det = '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(self.type,self.truncation,self.occlusion,self.alpha,\
                          self.xmin,self.ymin,self.xmax,self.ymax,self.h,self.w,self.l,self.t[0],self.t[1],self.t[2],self.ry)
        return self.label_line_det
    def revert_dontcare(self):
        self.org_type = self.type + ''
        self.type = 'DontCare'
        self.truncation = -1
        self.occlusion = -1
        self.alpha = -10.0
        self.h = -1000.0
        self.w = -1000.0
        self.l = -1000.0
        self.t= (-10.0,-1.0,-1.0)
        self.ry = -1.0
        
    def print_object(self):
        print('Frame, track id: %d, %d' % \
              (self.frame, self.track_id))
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
              (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
              (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
              (self.t[0], self.t[1], self.t[2], self.ry))
    def check_object_type(self):
        '''
        Determines the difficulty level of the given kitti object instance.
        '''
        if np.abs(self.ymin-self.ymax)>=40 and self.truncation<=0.15 and (self.occlusion in [0,3]):
            self.diff_level = 'easy'
        elif np.abs(self.ymin-self.ymax)>=25 and self.truncation<=0.3 and (self.occlusion in [0,1,3]):
            self.diff_level = 'moderate'
        elif np.abs(self.ymin-self.ymax)>=25 and self.truncation<=0.5:
            self.diff_level = 'hard'
        else:
            self.diff_level = 'unknown'


class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''

    def __init__(self, calib_filepath, from_video=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        try:
            self.V2C = calibs['Tr_velo_to_cam']
        except:
            self.V2C = calibs['Tr_velo_cam'] # Tracking dataset key           
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        try:
            self.R0 = calibs['R0_rect'] # For object detection dataset calib
        except:
            self.R0 = calibs['R_rect'] # For tracking dataset calib
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self,filepath):
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
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
    
        return data

    def read_calib_from_video(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        data = {}
        cam2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        Tr_velo_to_cam = np.zeros((3, 4))
        Tr_velo_to_cam[0:3, 0:3] = np.reshape(velo2cam['R'], [3, 3])
        Tr_velo_to_cam[:, 3] = velo2cam['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)


def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def transform_from_rot_trans(R, t):
    ''' Transforation matrix from rotation matrix and translation vector. '''
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def read_label(label_filename,from_video=False,frame=None):
    lines = [line.rstrip() for line in open(label_filename)]
    if from_video:
        if frame is not None:
            objects = []
            for line in lines:
                obj = Object3d_video(line)
                # Only add objects of this frame
                if obj.frame == frame:
                    objects.append(obj)
                else:
                    pass
            #objects = [Object3d_video(line) for line in lines]
        else:
            objects = [Object3d_video(line) for line in lines]
    else:
        objects = [Object3d(line) for line in lines]
    return objects


def load_image(img_filename):
    return cv2.imread(img_filename)


def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


def project_to_image(pts_3d, P):
    ''' Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def compute_box_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l;
    w = obj.w;
    h = obj.h;

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h];
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0];
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1];
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2];
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P);
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)


def compute_orientation_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        object orientation vector into the image plane.
        Returns:
            orientation_2d: (2,2) array in left image coord.
            orientation_3d: (2,3) array in in rect camera coord.
    '''

    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # orientation in object coordinate system
    orientation_3d = np.array([[0.0, obj.l], [0, 0], [0, 0]])

    # rotate and translate in camera coordinate system, project in image
    orientation_3d = np.dot(R, orientation_3d)
    orientation_3d[0, :] = orientation_3d[0, :] + obj.t[0]
    orientation_3d[1, :] = orientation_3d[1, :] + obj.t[1]
    orientation_3d[2, :] = orientation_3d[2, :] + obj.t[2]

    # vector behind image plane?
    if np.any(orientation_3d[2, :] < 0.1):
        orientation_2d = None
        return orientation_2d, np.transpose(orientation_3d)

    # project orientation into the image plane
    orientation_2d = project_to_image(np.transpose(orientation_3d), P);
    return orientation_2d, np.transpose(orientation_3d)


def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3 / CV_AA
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)
    return image

def tracking2objdet(tracking_path,drive_ids,output_path=None):
    '''
    Converts KITTI tracking label format into KITTI object detection format.
    tracking_path: Path to the KITTI tracking dataset. Inside: data_tracking_image_2, data_tracking_calib, data_tracking_label_2, data_tracking_velodyne folders
    drive_ids : List of drive ids whose labels will be converted. [0,1,2,3]
    output_path : Path showing where the generated labels will be saved. This defines the root path and each drive will be separately saved under a folder with the drive name: 0000/000000.txt,000001.txt,...; 0011/000000.txt,000001.txt,... If None, see below.
    
    Generates a new folder named as "drives_in_kitti" in the tracking_path if the output_path is None. Inside this folder, drive labels are separated according to their names.
    tracking_path
    ---drives_in_kitti
       ---0000
          ---000000.txt
          ---000001.txt
          ---*.txt
       0000_val.txt : A text file with the frame numbers of the labels in 0000
       ---0011
          ---000000.txt
          ---000001.txt
          ---*.txt
       0011_val.txt : A text file with the frame numbers of the labels in 0011
          
    
    '''
    if output_path is None:
        gt_label_path = os.path.join(tracking_path,'drives_in_kitti')
    else:
        gt_label_path = output_path
        if not os.path.exists(gt_label_path): os.mkdir(gt_label_path)
    #drive_paths = [os.path.join(gt_label_path,'{:04d}'.format(dr)) for dr in val_drives]
    for drive in drive_ids:
        # Create the drive folder
        drive_path = os.path.join(gt_label_path,'{:04d}'.format(drive))
        if os.path.isdir(drive_path):
            print("Drive folder exists!")
        else:
            os.makedirs(drive_path)

        ## Get the dataset instance of a specific drive with functions to get objects
        drive_dataset = get_dataset(video_num=drive,main_path=tracking_path)

        # Get image addresses to see how many images to go through
        img_addrs = drive_dataset.image_addresses
        str_val=''
        # Go through imgs to read objs of each
        for img_addr in img_addrs:
            # int image index
            img_id = drive_dataset.get_id_from_addr(img_addr)[0]
            # objects as instances of tracking_object class
            objs = drive_dataset.get_objects(ind=img_id,filter_out_classes=[])

            # Go through objects to generate string to be written into the file
            len_obj = len(objs)
            objs_str = ''
            for i_obj,obj in enumerate(objs):
                objs_str += '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(obj.class_name, obj.truncated,\
                                                                              obj.occluded, obj.alpha, obj.x1,\
                                                                              obj.y1, obj.x2, obj.y2,\
                                                                              obj.h, obj.w, obj.l, obj.x,\
                                                                              obj.y,obj.z,obj.rotation_y)

                if i_obj != len_obj-1:
                    objs_str += '\n'
                else:
                    pass
            # Open the file for each image and write labels
            with open(os.path.join(drive_path,'{:06d}.txt'.format(img_id)),'w') as file_name:
                file_name.write(objs_str)
            str_val+='{:06d}\n'.format(img_id)
            


        with open(os.path.join(gt_label_path,'val_{:04d}.txt'.format(drive)),'w') as file_name:
            file_name.write(str_val)
