import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import argrelextrema
sys.path.insert(0, "D:\Radi\Radi RU\4ti kurs\2sm-WBU\MOCAP\Python\mocap")
# import mocap_tools

import tools as t

if __name__ == "__main__":
	dictionary_path = ''
	###
	#
	# load whole take file with start/end frame without T-poses
	#
	###

	filename = os.path.join(dictionary_path, 'projevy_pocasi_01_ob_rh_lh_b_g_face_gaps.c3d')
	title = 'Pocasi_01'
	start_frame = 600
	end_frame = 7445

	# start_frame = 1000
	# end_frame = 1710


	# filename = os.path.join(dictionary_path, 'projevy_pocasi_02_ob_rh_lh_b_g_face_gaps.c3d')
	# title = 'Pocasi_02'
	# start_frame = 608
	# end_frame = 6667
	
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
	
	
	print("* * * {} * * *".format(title))


	###
	#
	# change origin point to be between the hip's markers // True is to find the relative coordinates
	#
	###
	new_origin = ['RFWT', 'LFWT', 'RBWT', 'LBWT']
	new_data = t.change_origin_point(data, new_origin, marker_list, True)
	
	###
	#
	## checks for dominant hand
	#
	##
	right_dominant = t.dominant_hand(start_frame, end_frame, data, marker_list)
	hand='R'
	if(right_dominant == 1):
		print("- more active hand is: Right \n")
	else:
		print("- more active hand is: Left hand \n")
		hand='L'


	###
	# compute velocity based on original trajectory 
	# returns - 3 chanel (x,y,z) velocity and normilized velocity
	###	
	r_velocity, r_vel = t.hand_velocity(start_frame, end_frame, new_data, marker_list, hand)
	# compute the median value vor velocity used as threshold for later analysis and segmentation
	median = np.median(r_vel)

	# compute acceleration based on normilized velocity
	r_acc = t.hand_acceleration(r_vel)
	# low pass filter of the acceleration for removing noise coused by recording techology
	r_acc_filt = t.butter_filter(r_acc, 12, fps, 10)
	
	###
	# raw segmentation of whole take
	# finding start and end frame of signs (exiting rest pose - entering rest pose)
	###	
	zero_crossing = t.zero_crossing(r_acc_filt)

	labels = t.segm(r_vel, r_acc_filt, start_frame, end_frame, new_data, marker_list, median)
	signs, signs1 = t.get_signs_borders(labels, start_frame, end_frame)


	count = len(signs)
	# analyze velocity during rest pose for better defining the threshold used for segmentation 
	rest_pose_vel = np.zeros([end_frame - start_frame])
	for i in range(0, count-1):
		st = signs[i][1]
		en = signs[i+1][0] 
		n = en-st

		vel, vel_norm = t.hand_velocity(st+start_frame,en+start_frame,new_data, marker_list, hand)
		rest_pose_vel[st:en] = vel_norm

	tr = np.amax(rest_pose_vel)
	print("threshold = ", tr)

	# refined starts and ends of the signs 
	labels2 = t.segm(r_vel, r_acc_filt, start_frame, end_frame, new_data, marker_list, tr)
	signs2, signs3 = t.get_signs_borders(labels2, start_frame, end_frame)
	# signs2, signs3 = t.segment_signs(start_frame, end_frame, new_data,marker_list,fps, tr)
	count2 = len(signs2)

	

	###
	#
	# analysis of each sign beased on refined raw segmentation 
	# 
	###	
	for sign in enumerate(signs2):
		st = sign[1][0]+start_frame
		en = sign[1][1]+start_frame

		print("*****")
		print("{}. The sign between {} - {} frames".format(sign[0]+1, st, en))
		

		vel_norm = r_vel[st-start_frame:en-start_frame]
		acc = r_acc[st-start_frame:en-start_frame]
		acc_f = r_acc_filt[st-start_frame:en-start_frame]
		
		zeros = t.zero_crossing(acc_f)
		zeros1 = t.interesting_points(acc_f)
		# print(zeros)
		# print(np.shape(zeros))

		# ## not OK
		# start = zeros[1]
		# end = zeros[np.shape(zeros)[0]-3]
		# print('Real start and end')
		# print("{}-{}".format(start+st, end+st))
		
		# print(zeros+start_frame)
		l = t.segm(vel_norm, acc_f, st-start_frame, en-start_frame, new_data, marker_list, tr)
		start1, end1 = t.get_real_signs(vel_norm, l, st, en )
		# print('Real start and end ')
		# print("{}-{}".format(start1+st, end1+st))
	
		print()
		print("Real start and end are {}-{}".format(start1+st, end1+st))

		diff_right_hand, diff_left_hand, diff_RWRE, diff_LWRE, right_dominant, one_hand = t.hand_displacment(start1+st, end1+st, new_data, marker_list)
		
		message = "The sign is \n-" 
		# "{}. The sign between {} - {} frames is".format(sign[0]+1, start1+st, end1+st)
		if(one_hand == 3):
			message = message + ' two handed'
		elif(one_hand == 1):
			message = message + ' right handed'
		elif(one_hand == 2):
			message = message + " left handed"
		else:
			message = message + " with no hands"

		print(message)
		h='R'
		if(right_dominant == 1):
			print("- dominant hand is: Right \n")
		else:
			print("- dominant hand is: Left hand \n")
			h='L'

		
		if(one_hand == 3):
			r_loc, ch_c_r = t.hand_location(start1+st, end1+st, new_data, marker_list, 'R') 
			l_loc, ch_c_l = t.hand_location(start1+st, end1+st, new_data, marker_list, 'L') 
			regions_r, count_r = np.unique(r_loc[:,[1]], return_counts = True)
			print("- Right hand changes in location: {}".format(ch_c_r))
			print("- Right hand is in:")
			# print(ch_c_r)
			# print(len(regions_r))
			for i in range(0, len(regions_r)):
				print("  - region {} for {} frames ".format(regions_r[i], count_r[i]))
			regions_l, count_l = np.unique(l_loc[:,[1]], return_counts = True)
			print("\n- Left hand changes in location: {}".format(ch_c_l))
			print("- Left hand is in:")
			# print(ch_c_l)
			# print(len(regions_l))
			for j in range(0, len(regions_l)):
				print(" - region {} for {} frames ".format(regions_l[j], count_l[j]))
		else:
			loc, ch_c = t.hand_location(start1+st, end1+st, new_data, marker_list, h) 
			regions, count = np.unique(loc[:,[1]], return_counts = True)
			print("- Dominant hand is in \n")
			for i in range(0, len(regions)):
				print("- region {} for {} frames ".format(regions[i], count[i]))

	
		# t.plot_hand_location(start1+st, end1+st, new_data, marker_list)

		x = np.arange(st, en)

		fig1 = plt.figure("{}_{}-{}-{}sign-vel".format(sign[0]+1,title,st,en), figsize=(10.5,7))
		fig1.suptitle("{}. Hand velocity for sign between {} and {} frame".format(sign[0]+1, st, en))	

		plt.plot(x, vel_norm, 'c', label='Normilized velocity') 
		# plt.plot(x, acc, 'm', label='Acc') 
		plt.plot(x, acc_f, 'y', label='Filtered acc') 
		# plt.plot(x[zeros1], vel_norm[zeros1], 'mo')
		plt.plot(x[zeros], vel_norm[zeros], 'o')

		# plt.plot(x[start], vel_norm[start], 'rs', label="Start")
		# plt.plot(x[end], vel_norm[end], 'r*', label= "End")

		plt.plot(x[start1], vel_norm[start1], 'gs', label="Start1")
		plt.plot(x[end1], vel_norm[end1], 'g*', label= "End1")
		

		plt.axhline(y=tr, color='r', linestyle='-', label="Treshold")
		plt.ylabel("Velocity (mm/frame)") 
		plt.xlabel("Frames")
		plt.grid(True)
		legend = fig1.legend(loc='upper right')
		# t.plot_hand_location(st, en, new_data, marker_list)
	plt.show()


	# print(signs[8][0]+start_frame)
	# t.plot_hand_location(signs[8][0]+start_frame, signs[8][1]+start_frame, new_data, marker_list)


