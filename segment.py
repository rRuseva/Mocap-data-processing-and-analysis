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
	
	# get sign borders
	signs, s1 = t.segment_signs(start_frame, end_frame, new_data, marker_list, fps)
	segm_data = t.get_segmented_data(start_frame, end_frame, new_data, marker_list, fps, 0)

	print(signs.shape)
	print(s1.shape)
	print(segm_data.shape)