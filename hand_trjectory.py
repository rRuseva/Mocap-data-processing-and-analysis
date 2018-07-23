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
	
	
	r_hand_tr, r_arm_tr = t.hand_trajectory(start_frame, end_frame, new_data, marker_list, 'R')
	r_h_tr, r_a_tr = t.norm_trajectory(r_hand_tr, r_arm_tr)
	
	l_hand_tr, l_arm_tr = t.hand_trajectory(start_frame, end_frame, new_data, marker_list, 'L')
	l_h_tr, l_a_tr = t.norm_trajectory(r_hand_tr, r_arm_tr)

	x = np.arange(start_frame, end_frame)

	fig1 = plt.figure("{}-R-Hand-trajectory-{}-{}".format(title, start_frame, end_frame), figsize=(10.5,7))
	fig1.suptitle("Right hand trajectory for sign between {} and {} frame".format(start_frame, end_frame))	

	plt.plot(x,r_hand_tr[:,[0]], 'r', label='x') 
	plt.plot(x,r_hand_tr[:,[1]], 'g', label='y') 
	plt.plot(x,r_hand_tr[:,[2]], 'b', label='z')

	plt.plot(x,r_h_tr, 'c', label='Normilized')
	
	plt.ylabel("Trajectory (mm)") 
	plt.grid(True)

	legend = fig1.legend(loc='upper right')
	
	fig2 = plt.figure("{}-L-Hand-trajectory-{}-{}".format(title, start_frame, end_frame), figsize=(10.5,7))
	fig2.suptitle("Left hand trajectory for sign between {} and {} frame".format(start_frame, end_frame))	

	plt.plot(x,l_hand_tr[:,[0]], 'r', label='x') 
	plt.plot(x,l_hand_tr[:,[1]], 'g', label='y') 
	plt.plot(x,l_hand_tr[:,[2]], 'b', label='z')
	plt.plot(x,l_h_tr, 'c', label='Normilized')
	
	plt.ylabel("Trajectory (mm)") 
	plt.grid(True)

	legend = fig2.legend(loc='upper right')

	plt.show()