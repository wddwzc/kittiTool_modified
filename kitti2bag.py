#!env python
# -*- coding: utf-8 -*-

import sys

import tf
import os
import cv2
import rospy
import rosbag
import progressbar
from tf2_msgs.msg import TFMessage
from datetime import datetime
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, Imu, PointField, NavSatFix
import sensor_msgs.point_cloud2 as pcl2
from geometry_msgs.msg import TransformStamped, TwistStamped, Transform, Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import numpy as np
import argparse

import utils
from odometry import odometry

import sometools as tools
from scipy.spatial.transform import Rotation as R


def save_dynamic_tf(bag, kitti, kitti_type, initial_time):
    print("Exporting time dependent transformations")
    timestamps = map(lambda x: initial_time + x.total_seconds(), kitti.timestamps)
    for timestamp, tf_matrix in zip(timestamps, kitti.T_w_cam0):
        tf_msg = TFMessage()
        tf_stamped = TransformStamped()
        tf_stamped.header.stamp = rospy.Time.from_sec(timestamp)
        tf_stamped.header.frame_id = 'world'
        tf_stamped.child_frame_id = 'camera_left'

        t = tf_matrix[0:3, 3]
        q = tf.transformations.quaternion_from_matrix(tf_matrix)
        transform = Transform()

        transform.translation.x = t[0]
        transform.translation.y = t[1]
        transform.translation.z = t[2]

        transform.rotation.x = q[0]
        transform.rotation.y = q[1]
        transform.rotation.z = q[2]
        transform.rotation.w = q[3]

        tf_stamped.transform = transform
        tf_msg.transforms.append(tf_stamped)

        bag.write('/tf', tf_msg, tf_msg.transforms[0].header.stamp)


# 不知道为啥转出来，camera_left_init can't find
def save_pose_tf(bag, kitti, from_frame_id, to_frame_id, topic):
	print("Exporting poses as TF")
	print(len(kitti.timestamps), len(kitti.poses))
	iterable = zip(kitti.timestamps, kitti.poses)
	bar = progressbar.ProgressBar()
	for dt, tf_pose in bar(iterable):
		tf_msg = TFMessage()

		tf_stamped = TransformStamped()
		tf_stamped.header.stamp = rospy.Time.from_sec(float(dt.strftime("%s.%f")))
		tf_stamped.header.frame_id = from_frame_id
		tf_stamped.child_frame_id = to_frame_id

		t = tf_pose[0:3, 3]
		q = tf.transformations.quaternion_from_matrix(tf_pose)
		transform = Transform()
		transform.translation.x = float(t[0])
		transform.translation.y = float(t[1])
		transform.translation.z = float(t[2])
		transform.rotation.x = float(q[0])
		transform.rotation.y = float(q[1])
		transform.rotation.z = float(q[2])
		transform.rotation.w = float(q[3])

		tf_stamped.transform = transform
		tf_msg.transforms.append(tf_stamped)

		bag.write(topic, tf_msg, tf_msg.transforms[0].header.stamp)


def save_laserpose_tf(bag, kitti, from_frame_id, to_frame_id, topic):
	print("Exporting poses as TF")
	print(len(kitti.timestamps), len(kitti.poses))
	iterable = zip(kitti.timestamps, kitti.poses)
	bar = progressbar.ProgressBar()

	origin_pose = inv(kitti.calib.T_cam0_velo).dot(kitti.poses[0])
	for dt, tf_pose in bar(iterable):
		tf_msg = TFMessage()

		tf_stamped = TransformStamped()
		tf_stamped.header.stamp = rospy.Time.from_sec(float(dt.strftime("%s.%f")))
		tf_stamped.header.frame_id = from_frame_id
		tf_stamped.child_frame_id = to_frame_id

		# pose本来是camera的位姿，转成了lidar坐标系下的位姿
		temp_t = inv(kitti.calib.T_cam0_velo).dot(tf_pose)
		temp_t[0:3, 3] -= origin_pose[0:3, 3]
 
		t = temp_t[0:3, 3]
		q = tf.transformations.quaternion_from_matrix(tf_pose)
		transform = Transform()
		transform.translation.x = float(t[0])
		transform.translation.y = float(t[1])
		transform.translation.z = float(t[2])
        # 转坐标系
		transform.rotation.x = float(q[2])
		transform.rotation.y = float(-q[0])
		transform.rotation.z = float(-q[1])
		transform.rotation.w = float(q[3])

		tf_stamped.transform = transform
		tf_msg.transforms.append(tf_stamped)

		bag.write(topic, tf_msg, tf_msg.transforms[0].header.stamp)


# 存所有的tf
def save_all_tf(bag, kitti, from_frame_id, to_frame_id, topic):
    print("Exporting all TF")
    print(len(kitti.timestamps), len(kitti.poses))
    iterable = zip(kitti.timestamps, kitti.poses)
    bar = progressbar.ProgressBar()

    for dt, tf_pose in bar(iterable):
        tf_msg = TFMessage()
        time = rospy.Time.from_sec(float(dt.strftime("%s.%f")))

        transform_calib = get_static_transform("camera_left", "velodyne", kitti.calib.T_cam0_velo)
        transform_calib.header.stamp = time
        tf_msg.transforms.append(transform_calib)
        transform_camera_left = get_static_transform("camera_left_init", "camera_left", tf_pose)
        transform_camera_left.header.stamp = time
        tf_msg.transforms.append(transform_camera_left)

        bag.write(topic, tf_msg, tf_msg.transforms[0].header.stamp)


def save_pose(bag, kitti, frame_id, topic):
    print("Exporting left camera poses(ground truth)")
    print (len(kitti.timestamps), len(kitti.poses))
    iterable = zip(kitti.timestamps, kitti.poses)
    bar = progressbar.ProgressBar()
    odom_list = []
    for timestamp, pose in bar(iterable):
        odom = Odometry()
        odom.header.frame_id = frame_id
        odom.child_frame_id = 'ground_truth'
        odom.header.stamp = rospy.Time.from_sec(float(timestamp.strftime("%s.%f")))
        t = pose[0:3, 3]
        q = tf.transformations.quaternion_from_matrix(pose)
        odom.pose.pose.position.x = t[0]
        odom.pose.pose.position.y = t[1]
        odom.pose.pose.position.z = t[2]
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        odom.twist.twist.linear.x = 0
        odom.twist.twist.linear.y = 0
        odom.twist.twist.angular.z = 0
        odom_list.append(odom)

        bag.write(topic, odom, t=odom.header.stamp)


def save_laserpose(bag, kitti, frame_id, topic):
    print("Exporting lidar poses(ground truth)")
    print (len(kitti.timestamps), len(kitti.poses))
    iterable = zip(kitti.timestamps, kitti.poses)
    bar = progressbar.ProgressBar()
    odom_list = []

    origin_pose = inv(kitti.calib.T_cam0_velo).dot(kitti.poses[0])
    for timestamp, pose in bar(iterable):
        odom = Odometry()
        odom.header.frame_id = frame_id
        odom.child_frame_id = 'ground_truth'
        odom.header.stamp = rospy.Time.from_sec(float(timestamp.strftime("%s.%f")))

        # 用标定信息对里程计进行了处理
        # pose 是 camera 的位姿，即相对 camera_init 的位姿
        # 但是轴选取的不一样
        # 但方向还是相对camera_init的，只是转到了lidar_init
        temp_t = inv(kitti.calib.T_cam0_velo).dot(pose)
        temp_t[0:3, 3] -= origin_pose[0:3, 3]
        
        t = temp_t[0:3, 3]
        q = tf.transformations.quaternion_from_matrix(pose)

        odom.pose.pose.position.x = t[0]
        odom.pose.pose.position.y = t[1]
        odom.pose.pose.position.z = t[2]
        # 把轴转成了lidar的坐标系
        odom.pose.pose.orientation.x = q[2]
        odom.pose.pose.orientation.y = -q[0]
        odom.pose.pose.orientation.z = -q[1]
        odom.pose.pose.orientation.w = q[3]
        odom.twist.twist.linear.x = 0
        odom.twist.twist.linear.y = 0
        odom.twist.twist.angular.z = 0
        odom_list.append(odom)

        #with open("/home/wzc/Data/kitti/data_odometry_laser/kittiTool_modified/kitti_poses_{}.txt".format(00), 'a') as f:
            # f.writelines([q[2], -q[0], -q[1], t[0], t[1], t[2]])
        #    ddddd = "{} {} {} {} {} {} {}\n".format(q[2], -q[0], -q[1], q[3], t[0], t[1], t[2])
        #    f.write(ddddd)

        bag.write(topic, odom, t=odom.header.stamp)


def save_velo_data(bag, kitti, velo_frame_id, topic):
    print("Exporting velodyne data")
    print (len(kitti.timestamps), len(kitti.velo_files))
    iterable = zip(kitti.timestamps, kitti.velo_files)
    bar = progressbar.ProgressBar()
    for dt, filename in bar(iterable):
        # if dt is None:
        #     continue

        # read binary data
        scan = (np.fromfile(filename, dtype=np.float32)).reshape(-1, 4)

        # create header
        header = Header()
        header.frame_id = velo_frame_id
        header.stamp = rospy.Time.from_sec(float(dt.strftime("%s.%f")))

        # fill pcl msg
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('intensity', 12, PointField.FLOAT32, 1)]
        pcl_msg = pcl2.create_cloud(header, fields, scan)

        bag.write(topic, pcl_msg, t=pcl_msg.header.stamp)


def save_label_data(bag, kitti, frame_id, topic):
    print("Exporting semantic label data")
    print (len(kitti.timestamps), len(kitti.label_files))
    iterable = zip(kitti.timestamps, kitti.label_files, kitti.velo_files)
    bar = progressbar.ProgressBar()
    for dt, label_filename, velo_filename in bar(iterable):
        # read binary data
        scan = (np.fromfile(velo_filename, dtype=np.float32)).reshape(-1, 4)
        label = (np.fromfile(label_filename, dtype=np.uint32)).reshape(-1)
        #print("##Print Shape##")
        #print(scan.shape)
        #print(label.shape)
        for i in range(label.shape[0] - 1):
            scan[i][3] = label[i] & 0xffff

        # create header
        header = Header()
        header.frame_id = frame_id
        header.stamp = rospy.Time.from_sec(float(dt.strftime("%s.%f")))

        # fill pcl msg
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('label', 12, PointField.UINT32, 1)]
        pcl_msg = pcl2.create_cloud(header, fields, scan)

        bag.write(topic, pcl_msg, t=pcl_msg.header.stamp)

def save_point_all(bag, kitti, frame_id, topic):
    print("Exporting semantic label data")
    print (len(kitti.timestamps), len(kitti.label_files))
    iterable = zip(kitti.timestamps, kitti.label_files, kitti.velo_files)
    bar = progressbar.ProgressBar()
    for dt, label_filename, velo_filename in bar(iterable):
        # read binary data
        scan = (np.fromfile(velo_filename, dtype=np.float32)).reshape(-1, 4)
        label = (np.fromfile(label_filename, dtype=np.uint32)).reshape(-1, 1)
        label = label & 0xffff

        #print("### Print Shape ###")
        #print(scan.shape)
        #print(label.shape)

        res_scan = np.concatenate((scan, label), axis=1)
        #print(res_scan.shape)
        
        # create header
        header = Header()
        header.frame_id = frame_id
        header.stamp = rospy.Time.from_sec(float(dt.strftime("%s.%f")))

        # fill pcl msg
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('intensity', 12, PointField.FLOAT32, 1),
                  PointField('label', 16, PointField.UINT32, 1)]
        pcl_msg = pcl2.create_cloud(header, fields, res_scan)

        bag.write(topic, pcl_msg, t=pcl_msg.header.stamp)


def get_static_transform(from_frame_id, to_frame_id, transform):
    t = transform[0:3, 3]
    q = tf.transformations.quaternion_from_matrix(transform)
    tf_msg = TransformStamped()
    tf_msg.header.frame_id = from_frame_id
    tf_msg.child_frame_id = to_frame_id
    tf_msg.transform.translation.x = float(t[0])
    tf_msg.transform.translation.y = float(t[1])
    tf_msg.transform.translation.z = float(t[2])
    tf_msg.transform.rotation.x = float(q[0])
    tf_msg.transform.rotation.y = float(q[1])
    tf_msg.transform.rotation.z = float(q[2])
    tf_msg.transform.rotation.w = float(q[3])
    return tf_msg


def inv(transform):
    "Invert rigid body transformation matrix"
    R = transform[0:3, 0:3]
    t = transform[0:3, 3]
    t_inv = -1 * R.T.dot(t)
    transform_inv = np.eye(4)
    transform_inv[0:3, 0:3] = R.T
    transform_inv[0:3, 3] = t_inv
    return transform_inv


# there are serveral calibration transformation, but only one left in this program
# so len(tfm.transforms) is 1
def save_static_transforms(bag, transforms, timestamps):
    print("Exporting static transformations")
    tfm = TFMessage()
    for transform in transforms:
        t = get_static_transform(from_frame_id=transform[0], to_frame_id=transform[1], transform=transform[2])
        tfm.transforms.append(t)
    for timestamp in timestamps:
        time = rospy.Time.from_sec(float(timestamp.strftime("%s.%f")))
        for i in range(len(tfm.transforms)):
            tfm.transforms[i].header.stamp = time
        # print(tfm)
        bag.write('/tf_static', tfm, t=time)

def main():
    parser = argparse.ArgumentParser(description = "Convert KITTI odometry dataset to ROS bag file the easy way!")
    odometry_sequences = []
    for s in range(11):
        odometry_sequences.append(str(s).zfill(2))
    parser.add_argument("-s", "--sequence", choices = odometry_sequences,help = "sequence of the odometry dataset (between 00 - 21), option is only for ODOMETRY datasets.")
    args = parser.parse_args()
    if args.sequence == None:
        print("Sequence option is not given.")
        sys.exit(1)

    velo_sequence = args.sequence
    print ('Start transmforming ' + velo_sequence)
    data_dir = "../"
    output_dir = os.path.join(data_dir, 'data_local', velo_sequence)
    output_file = os.path.join(output_dir, "kitti_odometry_{}.bag".format(velo_sequence))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(output_file):
        print('Bag has already exists')
        sys.exit(1)
    # topic info
    camera_left_init_frame_id = 'camera_left_init'
    camera_left_frame_id = 'camera_left'
    camera_left_topic = '/kitti/camera_left_poses'
    # velo_frame_id = 'velo_link'
    velo_frame_id = 'velodyne'
    velo_topic = '/kitti/velo/point'
    label_topic = '/kitti/velo/pointlabel'
    veloall_topic = '/kitti/velo/pointall'
    # rosbag settings
    compression = rosbag.Compression.NONE
    bag = rosbag.Bag(output_file, 'w', compression=compression)
    kitti = odometry(data_dir, velo_sequence)

    if not os.path.exists(kitti.sequence_path):
        print('Path {} does not exists. Exiting.'.format(kitti.sequence_path))
        sys.exit(1)

    # tf_static
    transforms = [
        # (camera_left_frame_id, velo_frame_id, inv(kitti.calib.T_cam0_velo))
        (camera_left_frame_id, velo_frame_id, kitti.calib.T_cam0_velo)
    ]

    try:
        # util = utils.read_calib_file(os.path.join(args.dir,'sequences',args.sequence, 'calib.txt'))
        # current_epoch = (datetime.utcnow() - datetime(1970, 1, 1)).total_seconds()
        
        # # Export
        # save_static_transforms(bag, transforms, kitti.timestamps)
        # save_all_tf(bag, kitti, camera_left_init_frame_id, camera_left_frame_id, "/tf_all")
        # save_pose(bag, kitti, camera_left_init_frame_id, camera_left_topic)
        # save_pose_tf(bag, kitti, camera_left_init_frame_id, camera_left_frame_id, "/tf_pose")
        # save_velo_data(bag, kitti, velo_frame_id, velo_topic)
        # save_label_data(bag, kitti, velo_frame_id, label_topic)

        #save_static_transforms(bag, transforms, kitti.timestamps)
        #save_pose(bag, kitti, camera_left_init_frame_id, camera_left_topic)
        save_laserpose(bag, kitti, "velodyne_init", "/kitti/velodyne_poses")
        #save_pose_tf(bag, kitti, camera_left_init_frame_id, camera_left_frame_id, "/tf_pose")
        #save_laserpose_tf(bag, kitti, "velodyne_init", velo_frame_id, "/tf_laserpose")
        
        save_velo_data(bag, kitti, velo_frame_id, velo_topic)
        save_label_data(bag, kitti, velo_frame_id, label_topic)
        
        # save_point_all(bag, kitti, velo_frame_id, veloall_topic)


    finally:
        print("## OVERVIEW ##")
        print(bag)
        bag.close()

# def _(arg):
#     pass

if __name__ == '__main__':
    main()
