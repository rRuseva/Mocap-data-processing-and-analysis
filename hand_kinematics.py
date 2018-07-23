import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.insert(0, "D:\Radi\Radi RU\4ti kurs\2sm-WBU\MOCAP\Python\mocap")
# import mocap_tools

import tools as t

if __name__ == "__main__":
	dictionary_path = ''
	
	filename = os.path.join(dictionary_path, 'projevy_pocasi_01_ob_rh_lh_b_g_face_gaps.c3d')
	title = 'Pocasi_01'
	start_frame = 600
	end_frame = 7445

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

	# filename = os.path.join(dictionary_path, 'projevy_pocasi_04_ob_rh_lh_b_g_face_gaps.c3d')
	# title = 'Pocasi_04'
	# start_frame = 600
	# end_frame = 5115

	data, marker_list, fps = t.read_frames(filename)
	
	# change origin point to be between the hip's markers // True is to find the relative coordinates
	new_origin = ['RFWT', 'LFWT', 'RBWT']
	new_data = t.change_origin_point(data, new_origin, marker_list, True)
	
	
	r_hand_tr, r_arm_tr = t.hand_trajectory(start_frame, end_frame, new_data, marker_list, 'R')
	
	r_velocity, r_vel = t.hand_velocity(start_frame, end_frame, new_data, marker_list, 'R')
	
	r_acc = t.hand_acceleration(r_vel)
	r_acc_filt = t.butter_filter(r_acc, 12, fps, 10)
	zero_crossing = t.zero_crossing(r_acc_filt)
	# extremums = t.find_extremums(r_vel)
	# median = np.median(r_vel)

	# labels = t.segm(r_vel, r_acc_filt, start_frame, end_frame, new_data, marker_list, median )
	# signs, signs1 = t.get_signs_borders(labels, start_frame, end_frame)
	# count = len(signs)

	x = np.arange(start_frame, end_frame)
	
	fig = plt.figure("{}-kinematics-{}-{}".format(title, start_frame, end_frame), figsize=(10.5,7))
	fig.suptitle("Right hand kinematics for sign between {} and {} frame".format(start_frame, end_frame))	

	plt.subplot(3, 1, 1)
	plt.plot(x,r_hand_tr[:,[0]], 'r', label='x') 
	plt.plot(x,r_hand_tr[:,[1]], 'g', label='y') 
	plt.plot(x,r_hand_tr[:,[2]], 'b', label='z')
	plt.ylabel("Trajectory (mm)") 
	plt.grid(True)

	plt.subplot(3, 1, 2)
	plt.plot(x,r_velocity[:,[0]], 'r') 
	plt.plot(x,r_velocity[:,[1]], 'g') 
	plt.plot(x,r_velocity[:,[2]], 'b')
	plt.plot(x, r_vel, 'm', label='Normilized velocity') 
	plt.plot(x[zero_crossing], r_vel[zero_crossing], 'o', label= "Extremums")
	# plt.plot(x[extremums], r_vel[extremums], 'o')
	# plt.plot(x[signs[:, 0]], r_vel[signs[:, 0]], 'rs', label = "Start")	
	# plt.plot(x[signs[:,1]], r_vel[signs[:,1]], 'r^', label = "End")	
	plt.ylabel("Velocity (mm/frame)") 
	plt.xlabel("Frames")
	plt.grid(True)

	plt.subplot(3, 1, 3)
	plt.plot(x,r_acc, 'c', label="Acceleration")
	plt.plot(x,r_acc_filt, 'm', label='Filtered acceleration')
	plt.plot(x[zero_crossing],r_acc_filt[zero_crossing], 'o')
	plt.ylabel("Acceleration over orig (mm/frame^2)")
	plt.xlabel("Frames")
	plt.grid(True)

	legend = fig.legend(loc='upper right')

	plt.show()