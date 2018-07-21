import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, "D:\Radi\Radi RU\4ti kurs\2sm-WBU\MOCAP\Python\mocap")
# import mocap_tools

import tools_new as t

if __name__ == "__main__":
	dictionary_path = ''
	
	# filename = os.path.join(dictionary_path, 'projevy_pocasi_01_ob_rh_lh_b_g_face_gaps.c3d')
	# title = 'Pocasi_01'
	# start_frame = 600
	# end_frame = 7445

	# start_frame = 1000
	# end_frame = 1710


	# filename = os.path.join(dictionary_path, 'projevy_pocasi_02_ob_rh_lh_b_g_face_gaps.c3d')
	# title = 'Pocasi_02'
	# start_frame = 608
	# end_frame = 6716
	
	# #end_frame = 1200
	# start_frame = 2500
	# end_frame = 3500
	
	# filename = os.path.join(dictionary_path, 'projevy_pocasi_03_ob_rh_lh_b_g_face_gaps.c3d')
	# title = 'Pocasi_03'
	# start_frame = 743
	# end_frame = 6680

	filename = os.path.join(dictionary_path, 'projevy_pocasi_04_ob_rh_lh_b_g_face_gaps.c3d')
	title = 'Pocasi_04'
	start_frame = 600
	end_frame = 5115

	data, marker_list, fps = t.read_frames(filename)
	
	# change origin point to be between the hip's markers // True is to find the relative coordinates
	new_origin = ['RFWT', 'LFWT', 'RBWT']
	new_data = t.change_origin_point(data, new_origin, marker_list, True)
		
	r_velocity, r_vel = t.hand_velocity(start_frame, end_frame, new_data, marker_list, 'R')
	
	r_acc = t.hand_acceleration(r_vel)
	r_acc_filt = t.butter_filter(r_acc, 5, fps, 10)
	zero_crossing = t.zero_crossing(r_acc_filt)
	
	signs, signs1 = t.segment_signs(start_frame, end_frame, new_data,marker_list,fps)
	count = len(signs)
	median = np.median(r_vel)

	print(count)
	print()
	print(new_data.shape)
	# rest_pose_vel = np.zeros([new_data.shape[0]])
	rest_pose_vel = np.zeros([end_frame - start_frame])
	# print(rest_pose_vel.shape)
	# rest_pose_vel = []

	j=0
	for i in range(0, count-1):
		print(i)
		st = signs[i][1]+start_frame
		en = signs[i+1][0] + start_frame
		n = en-st
		print(st, en)
		print(n)

		# print(new_data[st:en, :, :])
		vel, vel_norm = t.hand_velocity(st,en,new_data, marker_list,'R')
		print(vel_norm.shape)
		# print(rest_pose_vel[st:en].shape)
		# rest_pose_vel[st:en] = vel_norm
		
		rest_pose_vel[j:j+n] = vel_norm
		j = j+n
		print(j)
		print()

	# rest_pose_vel = np.delete(rest_pose_vel, np.s_[j:], 0)
	# rest_pose_vel = np.array(rest_pose_vel)
	print(rest_pose_vel)

	x = np.arange(start_frame, end_frame)

	fig1 = plt.figure("{}-rest-pose-vel".format(title), figsize=(10.5,7))
	fig1.suptitle("Right hand RP velocity for sign between {} and {} frame".format(start_frame, end_frame))	

	plt.plot(x, rest_pose_vel, 'c', label='Normilized velocity') 

	plt.axhline(y=median, color='r', linestyle='-', label="Treshold")
	plt.ylabel("Velocity (mm/frame)") 
	plt.xlabel("Frames")
	plt.grid(True)
	legend = fig1.legend(loc='upper right')

	plt.show()