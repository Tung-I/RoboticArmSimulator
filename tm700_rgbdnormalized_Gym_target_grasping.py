# Code base from pybullet examples https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/ kuka_diverse_object_gym_env.py


import random
import os
from gym import spaces
import time
import json
import pybullet as p
import numpy as np
import pybullet_data
import pdb
import distutils.dir_util
import glob
from pkg_resources import parse_version
import gym
from bullet.tm700 import tm700
from bullet.tm700_possensor_Gym import tm700_possensor_gym
# from bullet.grasping_Gym import tm700_rgbd_gym
from bullet.shapenet_gym import tm700_rgbd_gym
# from mayavi import mlab
from detection import inference

with open('./gripper_config.json', 'r') as fp:
    config = json.load(fp)
    config['thickness'] = 0.003

def get_name_to_link(model_id):
  link_name_to_index = {p.getBodyInfo(model_id)[0].decode('UTF-8'):-1,}
  for _id in range(p.getNumJoints(model_id)):
    _name = p.getJointInfo(model_id, _id)[12].decode('UTF-8')
    link_name_to_index[_name] = _id
  return link_name_to_index


if __name__ == '__main__':
  import sys
  import torch
  torch.backends.cudnn.benchmark = True
  torch.multiprocessing.set_start_method('forkserver')
  from gdn.representation.euler import *
  #from gdn.utils import *
  from gdn.detector.edgeconv.backbone import EdgeDet
  #from nms import nms as gripper_nms
  from nms import decode_euler_feature
  from nms import initEigen, sanity_check
  from nms import crop_index, generate_gripper_edge
  from scipy.spatial.transform import Rotation

  initEigen(0)

  output_path = sys.argv[2]
  assert output_path.endswith(('.txt', '.out', '.log'))
  total_n = int(sys.argv[3])

  gripper_length = config['hand_height']
  deepen_hand = gripper_length + 0.01
  model = EdgeDet(config, activation_layer=EulerActivation())
  model = model.cuda()
  model = model.eval()
  model.load_state_dict(torch.load(sys.argv[1])['base_model'])
  representation = EulerRepresentation(config)
  subsampling_util = val_collate_fn_setup(config)

  with open(output_path, 'w') as result_fp:
      p.connect(p.GUI)
      #p.setAdditionalSearchPath(datapath)
      start_obj_id = 3
      input_points = 2048
      ts = None #1/240.
      test = tm700_rgbd_gym(width=720, height=720, numObjects=7, objRoot='../YCB_valset_urdf')

      complete_n = 0
      max_tries = 3
      obj_success_rate = {}
      with torch.no_grad():
          for ite in range(total_n):
              test.reset()
              tm_link_name_to_index = get_name_to_link(test._tm700.tm700Uid)
              table_link_name_to_index = get_name_to_link(test.tableUid)

              obj_link_name_to_index = []
              for uid in test._objectUids:
                  obj_link_name_to_index.append((uid, get_name_to_link(uid)))

              object_set = list(test._objectUids)
              object_name = list(test._objNameList)
              grasp_success_obj = np.zeros(len(object_set), dtype=np.bool)
              grasp_failure_obj = np.zeros(len(object_set), dtype=np.int32)
              total_grasp_tries_count_down = max_tries * len(object_set)
              while total_grasp_tries_count_down>0 and (not grasp_success_obj.all()) and grasp_failure_obj.max()<max_tries:
                  total_grasp_tries_count_down -= 1
                  grasp_order = np.random.permutation(len(grasp_failure_obj)) # Randomly specify an object to grasp
                  for obj_i in grasp_order:
                      id_ = object_set[obj_i]
                      if grasp_success_obj[obj_i]:
                          continue
                      test._tm700.home()
                      # Clear out velocity of objects for consistancy
                      for _uid in test._objectUids:
                          p.resetBaseVelocity(_uid, [0, 0, 0], [0, 0, 0])

                      #####################################################################################
                      point_cloud, segmentation, rgb = test.getTargetGraspObservation()
                      #####################################################################################

                      obj_seg = segmentation.reshape(-1)==id_
                      if not obj_seg.any(): # Not visible now. Pick up another object to make it visible
                          print("Cant see anything. Next object!")
                          continue
                      pc_flatten = point_cloud.reshape(-1,3).astype(np.float32)
                      pc_no_arm = pc_flatten[segmentation.reshape(-1)>0,:] # (N, 3)
                      pc_npy = pc_flatten[obj_seg,:] # (N, 3)
                      while pc_npy.shape[0]<input_points:
                          new_pts = pc_npy[np.random.choice(len(pc_npy), input_points-len(pc_npy), replace=True)]
                          new_pts = new_pts + np.random.randn(*new_pts.shape) * 1e-6
                          pc_npy = np.unique(np.append(pc_npy, new_pts, axis=0), axis=0)
                      if pc_npy.shape[0]>input_points:
                          pc_npy = pc_npy[np.random.choice(len(pc_npy), input_points, replace=False)]
                      trans_to_frame = (np.max(pc_npy, axis=0) + np.min(pc_npy, axis=0)) / 2.0
                      trans_to_frame[2] = np.min(pc_npy[:,2])
                      pc_npy -= trans_to_frame

                      start_ts = time.time()
                      pc_batch, indices, reverse_lookup_index, _ = subsampling_util([(pc_npy,None),])
                      ss_ts = time.time()
                      print("Subsampling in %.2f seconds."%(ss_ts-start_ts))
                      try:
                          pred = model(pc_batch.cuda(), [pt_idx.cuda() for pt_idx in indices]).cpu().numpy()
                      except:
                          print("Prediction error!!! Please fix the bugs in preprocessing / model")
                          grasp_failure_obj[obj_i] += 1
                          if grasp_failure_obj[obj_i]>=max_tries:
                              break
                          else:
                              continue
                      inf_ts = time.time()
                      print("Inference in %.2f seconds."%(inf_ts-ss_ts))
                      feat_ts = time.time()
                      pred_poses = np.asarray(decode_euler_feature(
                            pc_npy[reverse_lookup_index[0]]+trans_to_frame[np.newaxis],
                            pred[0].reshape(1,-1),
                            *pred[0].shape[:-1],
                            config['hand_height'],
                            config['gripper_width'],
                            config['thickness_side'],
                            config['rot_th'],
                            config['trans_th'],
                            5000, # max number of candidate
                            -np.inf, # threshold of candidate
                            3000,  # max number of grasp in NMS
                            20,    # number of threads
                            True  # use NMS
                          ), dtype=np.float32)
                      print('Generated0 %d grasps.'%len(pred_poses))
                      filter_ts = time.time()
                      print("Decode in %.2f seconds."%(filter_ts-feat_ts))
                      pred_poses = sanity_check(pc_no_arm, pred_poses, 30,
                              config['gripper_width'],
                              config['thickness'],
                              config['hand_height'],
                              config['thickness_side'],
                              20 # num threads
                              )
                      end_ts = time.time()
                      print("Filter in %.2f seconds."%(end_ts-filter_ts))
                      print('Generated1 %d grasps in %.2f seconds.'%(len(pred_poses), end_ts-start_ts))
                      result_fp.write('Generated1 %d grasps in %.2f seconds.\n'%(len(pred_poses), end_ts-start_ts))
                      result_fp.flush()

                      new_pred_poses = []
                      for pose in pred_poses:
                          rotation = pose[:3,:3]
                          trans    = pose[:3, 3]
                          approach = rotation[:3,0]
                          if np.arccos(np.dot(approach.reshape(1,3), np.array([1, 0,  0]).reshape(3,1))) > np.radians(85):
                              continue
                          tmp_pose = np.append(rotation, trans[...,np.newaxis], axis=1)

                          # Sanity test
                          gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width']+config['thickness']*2,
                                                                                 config['hand_height'], tmp_pose,
                                                                                 config['thickness_side'], deepen_hand)
                          gripper_inner1, gripper_inner2 = generate_gripper_edge(config['gripper_width'], config['hand_height'],
                                                                                 tmp_pose, config['thickness_side'], 0.0)[1:]
                          outer_pts = crop_index(pc_no_arm, gripper_outer1, gripper_outer2)
                          inner_pts = crop_index(pc_no_arm[outer_pts], gripper_inner1, gripper_inner2)
                          gripper_l, gripper_r, gripper_l_t, gripper_r_t = gripper_inner_edge
                          if not (gripper_l_t[2] > -0.001 and gripper_r_t[2] > -0.001 and \
                                  gripper_l[2]   > -0.001 and gripper_r[2]   > -0.001 and \
                                  len(outer_pts) - len(inner_pts) < 30 and len(outer_pts) > 100):
                              continue

                          trans_backward = trans - approach * deepen_hand
                          tmp_pose = np.append(rotation, trans_backward[...,np.newaxis], axis=1)
                          gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width'],
                                                                                                     config['hand_height'],
                                                                                                     tmp_pose,
                                                                                                     config['thickness_side'],
                                                                                                     0.0
                                                                                                     )
                          gripper_l, gripper_r, gripper_l_t, gripper_r_t = gripper_inner_edge
                          if gripper_l_t[2] < -0.001 or gripper_r_t[2] < -0.001 or \
                             gripper_l[2]   < -0.001 or gripper_r[2]   < -0.001: # ready pose will collide with table
                              continue

                          new_pose = np.append(rotation, trans_backward[...,np.newaxis], axis=1)
                          new_pred_poses.append(new_pose)
                      pred_poses = new_pred_poses
                      print('Generated2 %d grasps'%len(pred_poses))
                      if len(pred_poses)==0:
                          print("No suitable grasp found.")
                          grasp_failure_obj[obj_i] += 1
                          if grasp_failure_obj[obj_i]>=max_tries:
                              break
                          else:
                              continue

                      tried_top1_grasp = None
                      for best_grasp in pred_poses:
                          rotation = best_grasp[:3,:3]
                          trans_backward = best_grasp[:,3]
                          approach = best_grasp[:3,0]
                          trans = trans_backward + approach*deepen_hand
                          pose = np.append(rotation, trans[...,np.newaxis], axis=1)
                          pose_backward = np.append(rotation, trans_backward[...,np.newaxis], axis=1)
                          for link_name, link_id in tm_link_name_to_index.items():
                              p.setCollisionFilterPair(test._tm700.tm700Uid, test.tableUid, link_id, -1, 0)
                              for obj_id, obj in obj_link_name_to_index:
                                  if obj_id in test._objectUids:
                                      for obj_name, obj_link in obj.items():
                                        # temporary disable collision detection and move to ready pose
                                        p.setCollisionFilterPair(test._tm700.tm700Uid, obj_id, link_id, obj_link, 0)
                          # Ready to grasp pose
                          test._tm700.home()
                          info = test.step_to_target_pose([pose, -0.0],  ts=ts, max_iteration=2000, min_iteration=1)[-1]
                          info_backward = test.step_to_target_pose([pose_backward, -0.0],  ts=ts, max_iteration=2000, min_iteration=1)[-1]
                          if tried_top1_grasp is None:
                              tried_top1_grasp = (pose_backward, pose)
                          if info['planning'] and info_backward['planning']:
                              break # Feasible Pose found
                          else:
                              print("Inverse Kinematics failed.")
                      if (not (info['planning'] and info_backward['planning'])) and (not tried_top1_grasp is None):
                          pose_backward, pose = tried_top1_grasp
                          test.step_to_target_pose([pose_backward, -0.0],  ts=ts, max_iteration=2000, min_iteration=1)[-1]
                      # Enable collision detection to test if a grasp is successful.
                      for link_name in ['finger_r_link', 'finger_l_link']:
                          link_id = tm_link_name_to_index[link_name]
                          for obj_id, obj in obj_link_name_to_index:
                              if obj_id in test._objectUids:
                                  for obj_name, obj_link in obj.items():
                                    p.setCollisionFilterPair(test._tm700.tm700Uid, obj_id, link_id, obj_link, 1)
                      # Enable collision detection for gripper head, fingers
                      #p.setCollisionFilterPair(test._tm700.tm700Uid, test.tableUid, tm_link_name_to_index['gripper_link'], -1, 1)
                      p.setCollisionFilterPair(test._tm700.tm700Uid, test.tableUid, tm_link_name_to_index['finger_r_link'], -1, 1)
                      p.setCollisionFilterPair(test._tm700.tm700Uid, test.tableUid, tm_link_name_to_index['finger_l_link'], -1, 1)
                      # Deepen gripper hand
                      for d in np.linspace(0, 1, 60):
                          info = test.step_to_target_pose([pose*d+pose_backward*(1.-d), -0.0],  ts=ts, max_iteration=100, min_iteration=1)[-1]
                      info = test.step_to_target_pose([pose, -0.0],  ts=ts, max_iteration=500, min_iteration=1)[-1]
                      if not info['planning']:
                          print("Inverse Kinematics failed.")
                      # Grasp it
                      test.step_to_target_pose([pose, 0.2],  ts=ts, max_iteration=500, min_iteration=5)
                      # Test if we can lift the object
                      pose[2,3] = 0.5
                      test.step_to_target_pose([pose, 0.2],  ts=ts, max_iteration=5000, min_iteration=5)
                      for _ in range(1000):
                          p.stepSimulation()
                      # Compute success rate for each object
                      this_grasp_success = False
                      if test.check_if_grasp_success(0.50, id_):
                          print("Grasp success!")
                          this_grasp_success = True
                      else:
                          print("Grasp failed!")
                          if False:
                              pc_subset = np.copy(pc_no_arm)
                              if len(pc_subset)>5000:
                                  pc_subset = pc_subset[np.random.choice(len(pc_subset), 5000, replace=False)]
                              mlab.clf()
                              mlab.points3d(pc_subset[:,0], pc_subset[:,1], pc_subset[:,2], scale_factor=0.004, mode='sphere', color=(1.0,1.0,0.0), opacity=1.0)
                              for n, pose_ in enumerate(pred_poses):
                                  pose = np.copy(pose_)
                                  pose[:,3] += pose[:,0] * deepen_hand
                                  gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width'], config['hand_height'], pose, config['thickness_side'], 0.0)
                                  gripper_l, gripper_r, gripper_l_t, gripper_r_t = gripper_inner_edge

                                  mlab.plot3d([gripper_l[0], gripper_r[0]], [gripper_l[1], gripper_r[1]], [gripper_l[2], gripper_r[2]], tube_radius=config['thickness']/4., color=(0,0,1) if n>0 else (1,0,0), opacity=0.5)
                                  mlab.plot3d([gripper_l[0], gripper_l_t[0]], [gripper_l[1], gripper_l_t[1]], [gripper_l[2], gripper_l_t[2]], tube_radius=config['thickness']/4., color=(0,0,1) if n>0 else (1,0,0), opacity=0.5)
                                  mlab.plot3d([gripper_r[0], gripper_r_t[0]], [gripper_r[1], gripper_r_t[1]], [gripper_r[2], gripper_r_t[2]], tube_radius=config['thickness']/4., color=(0,0,1) if n>0 else (1,0,0), opacity=0.5)
                              mlab.show()
                              input()
                      if this_grasp_success:
                          test._objectUids.remove(id_)
                          p.removeBody(id_)
                          grasp_success_obj[obj_i] = True
                          if grasp_success_obj.all():
                              break
                      else:
                          grasp_failure_obj[obj_i] += 1
                          if grasp_failure_obj[obj_i]>=max_tries:
                              break
                  if grasp_success_obj.all():
                      complete_n += 1
                  for obj_i in range(len(object_set)):
                      name = object_name[obj_i]
                      fail_n = grasp_failure_obj[obj_i]
                      success_n = 1 if grasp_success_obj[obj_i] else 0
                      if not name in obj_success_rate:
                          obj_success_rate[name] = (0, 0) # success , fail
                      success_n_old, fail_n_old = obj_success_rate[name]
                      obj_success_rate[name] = (success_n+success_n_old, fail_n+fail_n_old)
                  for obj_name, (success_n, fail_n) in obj_success_rate.items():
                      total_grasp_obj = success_n + fail_n
                      if total_grasp_obj==0:
                          result_fp.write("%s : Unknown\n"%obj_name)
                      else:
                          result_fp.write("%s : %.4f (%d / %d)\n"%(obj_name, success_n/total_grasp_obj, success_n, total_grasp_obj))
                      result_fp.flush()
              result_fp.write("Complete rate: %.6f (%d / %d)\n"%(complete_n/(ite+1), complete_n, ite+1))
              result_fp.flush()
