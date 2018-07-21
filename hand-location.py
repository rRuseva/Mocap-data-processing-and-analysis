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
	
	r_loc, c1 = t.hand_location(start_frame, end_frame, data, marker_list, h='R') 
	l_loc, c2 = t.hand_location(start_frame, end_frame, data, marker_list, h='L') 
	rp = t.rest_pose(start_frame, end_frame, data, marker_list) 
	

	x = np.arange(start_frame,end_frame)

	fig = plt.figure("{}-HandsLocation-{}-{}".format(title, start_frame, end_frame), figsize=(10.5,7))
	fig.suptitle("Hands location for sign between {} and {} frame".format(start_frame, end_frame))	
	
	# plt.subplot(2, 1, 1)
	plt.plot(x,r_loc[:,[1]], 'r', label='Right hand') 
	plt.plot(x,l_loc[:,[1]], 'g', label='Left hand') 
	plt.plot(x[rp],r_loc[:,[1]][rp], 'bo', label='Rest pose') 
	plt.grid(True)
	plt.xlabel("Frames")	
	plt.ylabel("Regions")

	# plt.subplot(2, 2, 3)
	# plt.title("R-hand location changes: {}".format(c1))
	# plt.hist(r_loc[:,[1]], bins=range(15), facecolor='r', align="left")
	# plt.xticks(np.arange(1, 16, step=1)) 
	# plt.ylabel("Number of Frames")	
	# plt.xlabel("Regions")

	# plt.subplot(2, 2, 4)
	# plt.title("L-hand location changes: {}".format(c2))
	# plt.hist(l_loc[:,[1]], bins=range(15),  facecolor='g', align="left") 
	# plt.xticks(np.arange(1, 16, step=1)) 
	# plt.ylabel("Number of Frames")	
	# plt.xlabel("Regions")
	
	legend = fig.legend(loc='upper right')
	plt.show()
	



