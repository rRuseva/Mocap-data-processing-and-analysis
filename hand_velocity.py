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
	
	
	# r_hand_tr, r_arm_tr = t.hand_trajectory(start_frame, end_frame, new_data, marker_list, 'R')
	
	r_velocity, r_vel = t.hand_velocity(start_frame, end_frame, new_data, marker_list, 'R')
	median = np.median(r_vel)

	r_acc = t.hand_acceleration(r_vel)
	r_acc_filt = t.butter_filter(r_acc, 12, fps, 10)
	zero_crossing = t.zero_crossing(r_acc_filt)


	labels = t.segm(r_vel, r_acc_filt, start_frame, end_frame, new_data, marker_list, median)
	signs, s1 = t.get_signs_borders(labels, start_frame, end_frame)
	count = len(signs)
	

	# analyze velocity during rest pose for better defining the threshold used for segmentation 
	rest_pose_vel = np.zeros([end_frame - start_frame])
	for i in range(0, count-1):
		st = signs[i][1]
		en = signs[i+1][0] 
		n = en-st

		vel, vel_norm = t.hand_velocity(st+start_frame,en+start_frame,new_data, marker_list,'R')
		rest_pose_vel[st:en] = vel_norm

	tr = np.amax(rest_pose_vel)
	print("threshold=", tr)

	# refined starts and ends of the signs 
	signs2, signs3 = t.segment_signs(start_frame, end_frame, new_data,marker_list,fps, tr)
	count2 = len(signs2)



	x = np.arange(start_frame, end_frame)

	fig1 = plt.figure("{}-{}signs-vel".format(title, count), figsize=(10.5,7))
	fig1.suptitle("Right hand velocity for sign between {} and {} frame".format(start_frame, end_frame))	

	plt.plot(x, r_vel, 'c', label='Normilized velocity') 
	plt.plot(x[zero_crossing], r_vel[zero_crossing], 'o')
	# startandend after 1st segm
	plt.plot(x[signs[:, 0]], r_vel[signs[:, 0]], 'rs', label = "Start 1")	
	plt.plot(x[signs[:, 1]], r_vel[signs[:, 1]], 'r*', label = "End 1")	
	plt.axhline(y=median, color='r', linestyle='-', label="Treshold")
	# startandend after 2nd segm
	# plt.plot(x[signs2[:, 0]], r_vel[signs2[:, 0]], 'ms', label = "Start 2")	
	# plt.plot(x[signs2[:, 1]], r_vel[signs2[:, 1]], 'm*', label = "End 2")
	plt.axhline(y=tr, color='m', linestyle='-', label="Refined Treshold ")	

	#real start and end
	# plt.plot(x[s1[:, 0]], r_vel[s1[:, 0]], 'gs', label = "Start")	
	# plt.plot(x[s1[:, 1]], r_vel[s1[:, 1]], 'g*', label = "End")

	plt.ylabel("Velocity (mm/frame)") 
	plt.xlabel("Frames")
	plt.grid(True)
	legend = fig1.legend(loc='upper right')


	fig2 = plt.figure("{}-{}signs-acc".format(title, count), figsize=(10.5,7))
	fig2.suptitle("Right hand acceleration for sign between {} and {} frame".format(start_frame, end_frame))	

	plt.plot(x, r_acc, 'c', label=' acceleration') 
	plt.plot(x, r_acc_filt, 'm', label='Filtered acceleration') 
	plt.plot(x[zero_crossing], r_acc_filt[zero_crossing], 'o')
	plt.plot(x[signs[:, 0]], r_acc_filt[signs[:, 0]], 'rs', label = "Start")	
	plt.plot(x[signs[:,1]], r_acc_filt[signs[:,1]], 'r^', label = "End")	
	plt.ylabel("Acceleration (mm/frame^2)") 
	plt.xlabel("Frames")
	plt.grid(True)
	legend = fig2.legend(loc='upper right')
	
	# plt.hist(r_vel, bins='auto') 
	plt.show()

	
