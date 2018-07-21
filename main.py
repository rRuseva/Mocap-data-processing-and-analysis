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
	start_frame = 600
	end_frame = 7500

	# start_frame = 2675
	# end_frame = 3640


	# filename = os.path.join(dictionary_path, 'projevy_pocasi_02_ob_rh_lh_b_g_face_gaps.c3d')
	# start_frame = 608
	# end_frame = 6716
	
	# #end_frame = 1200
	# start_frame = 2500
	# end_frame = 3500
	
	# filename = os.path.join(dictionary_path, 'projevy_pocasi_03_ob_rh_lh_b_g_face_gaps.c3d')
	# start_frame = 743
	# end_frame = 6680

	# filename = os.path.join(dictionary_path, 'projevy_pocasi_04_ob_rh_lh_b_g_face_gaps.c3d')
	# start_frame = 600
	# end_frame = 5115

	data, marker_list, fps = t.read_frames(filename)
	print(fps)
	# print(data[730:996, 1, :])
	# change origin point to be between the hip's markers // True is to find the relative coordinates
	new_origin = ['RFWT', 'LFWT', 'RBWT']
	new_data = t.change_origin_point(data, new_origin, marker_list, True)
	
	# print(new_data[730:1100, 0, :])
	# t.plot_hand(start_frame, end_frame, new_data, marker_list, fps)
	# t.plot_handsigns_boundaries_trajectory(start_frame, end_frame, new_data,marker_list)
	# t.plot_hand(start_frame, end_frame, new_data, marker_list, fps)
	# t.segment_signs(start_frame, end_frame, new_data, marker_list, fps)
	# t.plot_hands_location(start_frame, end_frame, new_data, marker_list)
	# t.plot_segmentation(start_frame, end_frame, new_data, marker_list, fps)
	
	segm_data = t.get_segmented_data(start_frame,end_frame, new_data,marker_list, fps)	
	
	n = np.shape(segm_data)[0]
	# print(np.shape(segm_data))
	# print()
	# print(np.shape(segm_data))
	# print(segm_data[0:270, 0, :])
	# for frame in range(0,n):	
	# 	print("{}-{}".format(frame,segm_data[frame, 0, :]))

	# t.plot_hand(start_frame, end_frame, new_data, marker_list, fps)

	
	# t.plot_hand(0, n-1, segm_data, marker_list, fps)
	# t.plot_hand2(start_frame, end_frame, new_data, marker_list, fps, "Original data")
	# t.plot_hand2(0, n-1, segm_data, marker_list, 120.0, "Segmented data")
	