import numpy as np
from numpy import linalg

import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import sys
import c3d
sys.path.insert(0, "D:\Radi\Radi RU\4ti kurs\2sm-WBU\MOCAP\Python\mocap")
import mocap_tools
import vector_math as vm
from scipy.signal import argrelextrema
import scipy.signal as signal

dictionary_path = ''
# filename = os.path.join(dictionary_path, 'projevy_pocasi_01_ob_rh_lh_b_g_face_gaps.c3d')
mlist_filename = 'marker_list_v0.txt'


def marker_list_read(filename):
	m_list = []
	with open(filename, 'r') as w:
		for line in w:
			m_list.append(line.split('\n')[0])

	return m_list

# mlist = marker_list_read(dictionary_path + mlist_filename)

def marker_index(mlist, mname):
	for i, m in enumerate(mlist):
		if(m == mname):
			return i
	
	return False

def marker_name(mlist, index):
	if(index < len(mlist)):
		return mlist[index]
	else: 
		return False
	
def read_frames(filename):

	with open(filename, 'rb') as _handle:
		reader = c3d.Reader(_handle)
		tot_frames = reader.last_frame() - reader.first_frame() + 1
		tot_markers = reader.point_used
		fps =  int(reader.header.frame_rate)
		
		# units = reader.get_string('POINT.UNITS')
		# print(units)
		# origin = reader.get('FORCE_PLATFORM.ORIGIN')
		# print(origin.float_array)
		
		# print(fps)
		# print(tot_markers)

		data = np.zeros((tot_frames, tot_markers+1, 5))
		for frame, (point) in enumerate(reader.read_frames()):
			for marker in range(tot_markers):
				# frame_index, marker_index, x,y,z,_,_
				data[frame, marker, :] = point[1][marker]/10	#point[0] is frame number
				#print(point[1][marker])
			data[frame, tot_markers,:] = np.zeros(5)
		#print(data[1:6, 0:1, 0:3])
		
		# print(reader.point_labels)
		makrer_list = []
		for marker in enumerate(reader.point_labels):
			makrer_list.append(marker[1].strip())
			#print(marker)
		makrer_list.append("ORIGIN")
		tot_markers = tot_markers+1

		# print(makrer_list)

		# # print(data[1:2, :, :])
		
		# print(marker_name(mlist,62))

		# print(marker_name(makrer_list,0))
		# print(marker_name(makrer_list,tot_markers-1))
		# m = marker_index(makrer_list, 'ORIGIN')
		# print(data[1, m, 0:3])

	return data[:, :, 0:3], makrer_list, fps

# change the origin point of a 'data' set to an absolute coordinates with center new_origin[] 
# or to return to an absolute coordinates if 'relative' param is false
def change_origin_point(data, new_origin, mlist, relative=True):
	new_data = np.zeros((int(data.shape[0]), int(data.shape[1]), int(data.shape[2])) )
	new_zero = 0
	for frame, (point) in enumerate(data):
		for marker in range(data.shape[1]):
			if(relative):
				for origin_marker in new_origin:
					new_zero = new_zero + data[frame, marker_index(mlist, origin_marker), :]
				new_zero = new_zero/len(new_origin)
				new_data[frame, marker,:] = data[frame, marker, :] - new_zero
			else:
				origin = data[frame, marker_index(mlist, 'ORIGIN'), :]
				new_data[frame, marker,:] = data[frame, marker, :] - origin
	
	return new_data

def hand_marker(data, mlist, h='R'):
	#artificial marker for 
	if(h == 'L'):
		OHAND_indx = marker_index(mlist, 'LOHAND')
		IHAND_indx = marker_index(mlist, 'LIHAND')
		IWR_indx = marker_index(mlist, 'LIWR')
		OWR_indx = marker_index(mlist, 'LOWR')

	else:
		OHAND_indx = marker_index(mlist, 'ROHAND')
		IHAND_indx = marker_index(mlist, 'RIHAND')
		IWR_indx = marker_index(mlist, 'RIWR')
		OWR_indx = marker_index(mlist, 'ROWR')
	
	OHAND = data[:, OHAND_indx, :]
	IHAND = data[:, IHAND_indx,:]
	IWR = data[:, IWR_indx,:]
	OWR = data[:, OWR_indx, :]
	
	hand = np.average([OHAND,IHAND,IWR,OWR], axis=0)
	return hand

def is_one_handed(start_frame, end_frame, data, mlist):
	differ = 1
	tot_frames = end_frame - start_frame
	# print(tot_frames)
	temp_data = data[start_frame:end_frame+1, :, :]
	
	is_right_handed = False
	is_left_handed = False

	#markers of interest
	ROWR_indx = marker_index(mlist, 'ROWR')				# ROWR and RIHAND are used to compute an artifical
	RIHAND_indx = marker_index(mlist, 'RIHAND')			# middle point on the upper side of the hand
	
	RWRE_indx = marker_index(mlist, 'RWRE')				# arm

	LOWR_indx = marker_index(mlist, 'LOWR')
	LIHAND_indx = marker_index(mlist, 'LIHAND')
	
	LWRE_indx = marker_index(mlist, 'LWRE')

	diff_right_hand = np.zeros([tot_frames, 3])
	diff_right_hand_sum = 0
	diff_RWRE = np.zeros([tot_frames, 3])
	diff_RWRE_sum = 0

	diff_left_hand = np.zeros([tot_frames, 3])
	diff_left_hand_sum = 0 
	diff_LWRE = np.zeros([tot_frames, 3])
	diff_LWRE_sum = 0

	# r_hand = (temp_data[0, ROWR_indx, :] + temp_data[0, RIHAND_indx, :])/2
	# r_hand_n = (temp_data[1, ROWR_indx, :] + temp_data[1, RIHAND_indx, :])/2
	# print((temp_data[1, ROWR_indx, :] + temp_data[1, RIHAND_indx, :])/2 - (temp_data[0, ROWR_indx, :] + temp_data[0, RIHAND_indx, :])/2)
	for frame in range(1, tot_frames, 1):
		# compute the center point of the right/ left hand 
		# print("ROWR-f:{}-{}".format(frame,temp_data[frame-1, ROWR_indx, :]))
		r_hand = (temp_data[frame-1, ROWR_indx, :] + temp_data[frame-1, RIHAND_indx, :])/2
		r_hand_n = (temp_data[frame, ROWR_indx, :] + temp_data[frame, RIHAND_indx, :])/2
		diff_right_hand[frame] = abs(r_hand_n - r_hand)
		diff_right_hand_sum = diff_right_hand_sum + diff_right_hand[frame]
		# print("{} + {} = {}".format(temp_data[frame-1, ROWR_indx, :], temp_data[frame-1, RIHAND_indx, :],r_hand))
		# print("{} + {} = {}".format(temp_data[frame, ROWR_indx, :], temp_data[frame, RIHAND_indx, :],r_hand_n))
		# print(diff_right_hand[frame])

		l_hand = (temp_data[frame-1, LOWR_indx, :] + temp_data[frame-1, LIHAND_indx, :])/2
		l_hand_n = (temp_data[frame, LOWR_indx, :] + temp_data[frame, LIHAND_indx, :])/2
		diff_left_hand[frame] = abs(l_hand_n - l_hand)
		diff_left_hand_sum = diff_left_hand_sum + diff_left_hand[frame]

		diff_RWRE[frame] = abs(temp_data[frame, RWRE_indx, :] - temp_data[frame-1, RWRE_indx, :])
		diff_RWRE_sum = diff_RWRE_sum + diff_RWRE[frame]
		
		diff_LWRE[frame] = abs(temp_data[frame, LWRE_indx, :] - temp_data[frame-1, LWRE_indx, :])
		diff_LWRE_sum = diff_LWRE_sum + diff_LWRE[frame]

		# print("{} {}".format(diff_left_hand_sum, diff_LWRE_sum))

	diff_right_hand_sum = diff_right_hand_sum/tot_frames
	diff_left_hand_sum = diff_left_hand_sum/tot_frames
	diff_RWRE_sum = diff_RWRE_sum/tot_frames
	diff_LWRE_sum = diff_LWRE_sum/tot_frames
	
	differ = ((diff_right_hand_sum + diff_left_hand_sum + diff_RWRE_sum + diff_LWRE_sum)/4)*9/10
	# print(differ)
	# print()
	# print("R h:{} a:{}".format(diff_right_hand_sum, diff_RWRE_sum))
	# print("L h:{} a:{}".format(diff_left_hand_sum, diff_LWRE_sum))
	if((abs(diff_right_hand_sum) > differ).any() or (abs(diff_RWRE_sum) > differ).any()):
		is_right_handed = True
		# print("r h {}".format((abs(diff_right_hand) > differ).any()))
		# print("rwre {}".format((abs(diff_RWRE) > differ).any()))
		
	if((abs(diff_left_hand_sum) > differ).any() or (abs(diff_LWRE_sum) > differ).any()):
		is_left_handed = True
		# print("l h {}".format((abs(diff_left_hand) > differ).any()))
		# print("lwre {}".format((abs(diff_LWRE) > differ).any()))

	# print("R {} {}".format(diff_right_hand_sum, diff_RWRE_sum))
	# print("L {} {}".format(diff_left_hand_sum, diff_LWRE_sum))
	# print()

	one_hand = 0 
	message = "The sign between {} - {} frames is".format(start_frame, end_frame)
	if(is_left_handed and is_right_handed):
		message = message + ' two handed'
		one_hand = 3
	elif(is_right_handed):
		message = message + ' right handed'
		one_hand = 1
	elif(is_left_handed):
		message = message + " left handed"
		one_hand = 2
	else:
		message = message + " with no hands"
		one_hand = 0

	if((diff_right_hand_sum > diff_left_hand_sum).any() or (diff_RWRE_sum > diff_LWRE_sum).any()):
		print("Dominant hand is: Right \n")
		right_dominant = 1
	else:
		print("Dominant hand is: Left hand \n")
		right_dominant = 0
	
	# plot_hand_movement(diff_right_hand, diff_left_hand, diff_RWRE, diff_LWRE, start_frame, end_frame, message)

	return diff_right_hand, diff_left_hand, diff_RWRE, diff_LWRE, right_dominant, message

def dominant_hand(start_frame, end_frame, data, mlist):
	diff_right_hand, diff_left_hand, diff_RWRE, diff_LWRE, right_dominant, m = is_one_handed(start_frame, end_frame, data, mlist)
	return right_dominant

def plot_hand_movement(start_frame, end_frame, data, mlist):
	diff_right_hand, diff_left_hand, diff_RWRE, diff_LWRE, right_dominant, message = is_one_handed(start_frame, end_frame, data, mlist)
	
	tot_fr = end_frame - start_frame
	x = np.arange(start_frame,end_frame ) 
	
	fig = plt.figure("HandsMovement", figsize=(12.5,7))
	fig.suptitle(message)
	
	plt.subplot(2, 2, 1)
	plt.title("Right hand movement")
	plt.grid(True)

	z_min = argrelextrema(diff_right_hand[:,[2]], np.greater, order=10)
	z_max = argrelextrema(diff_right_hand[:,[2]], np.less, order=10)

	plt.plot(x,diff_right_hand[:,[0]], 'r', label='x') 
	plt.plot(x,diff_right_hand[:,[1]], 'g', label='y') 
	plt.plot(x,diff_right_hand[:,[2]], 'b', label='z') 
	plt.plot(x[z_min[0]], diff_right_hand[:,[2]][z_min[0]], 'o', label='min')
	plt.plot(x[z_max[0]], diff_right_hand[:,[2]][z_max[0]], 'o', label='max')
	plt.xlabel("{} frames".format(tot_fr)) 	
	plt.ylabel("movement") 

	plt.subplot(2, 2, 2)
	plt.title("Left hand movement")
	plt.grid(True)

	plt.plot(x,diff_left_hand[:, [0]], 'r') 
	plt.plot(x,diff_left_hand[:, [1]], 'g') 
	plt.plot(x,diff_left_hand[:, [2]], 'b') 
	plt.xlabel("{} frames".format(tot_fr)) 	
	plt.ylabel("movement") 

	plt.subplot(2, 2, 3)
	plt.title("RWRE movement")
	plt.grid(True)

	plt.plot(x,diff_RWRE[:,[0]], 'r') 
	plt.plot(x,diff_RWRE[:,[1]], 'g') 
	plt.plot(x,diff_RWRE[:,[2]], 'b') 
	plt.xlabel("{} frames".format(tot_fr)) 	
	plt.ylabel("movement") 

	plt.subplot(2, 2, 4)
	plt.title("LWRE movement")
	plt.grid(True)

	plt.plot(x,diff_LWRE[:,[0]], 'r') 
	plt.plot(x,diff_LWRE[:,[1]], 'g') 
	plt.plot(x,diff_LWRE[:,[2]], 'b') 
	plt.xlabel("{} frames".format(tot_fr)) 	
	plt.ylabel("movement") 

	legend = fig.legend(loc='upper right')
	plt.show()

def hand_trajectory(start_frame, end_frame, data, mlist, h='R'):
	temp_data = data[start_frame:end_frame, :, :]

	hand_pos = np.zeros([end_frame-start_frame, 3])
	arm_pos = np.zeros([end_frame-start_frame, 3])
	
	hand_pos = hand_marker(temp_data, mlist, h)
	
	if(h == 'L'):
		arm_pos = temp_data[:, marker_index(mlist, 'LWRE'), :]
	else:
		arm_pos  = temp_data[:, marker_index(mlist, 'RWRE'), :]	

	return hand_pos, arm_pos

def zero_crossing(data):
	zero_crossing = []
	n = len(data)
	count = 0 
	for i in range(0, n-1):
		if((data[i]>0 and data[i+1]<0) or (data[i]<0 and data[i+1]>0)):
			zero_crossing.append(i)
			count= count + 1

	zero_crossing = np.array(zero_crossing)
	
	return zero_crossing
# normilize the hand and arm trajectory by the 3 axises
def norm_trajectory(hand_pos, arm_pos):
	hand = np.linalg.norm(hand_pos, axis=1)
	arm = np.linalg.norm(arm_pos, axis=1)

	return(hand, arm)

def plot_hand_trajectory(start_frame, end_frame, data, mlist):
	tot_fr = end_frame - start_frame
	x = np.arange(start_frame,end_frame ) 

	r_hand, r_arm = hand_trajectory(start_frame, end_frame, data, mlist, 'R')
	l_hand, l_arm = hand_trajectory(start_frame, end_frame, data, mlist, 'L')
	
	fig = plt.figure("HandsTrajectory", figsize=(12.5,7))
		
	plt.subplot(2, 2, 1)
	plt.title("Right hand Trajectory")
	plt.grid(True)

	plt.plot(x,r_hand[:,[0]], 'r', label='x') 
	plt.plot(x,r_hand[:,[1]], 'g', label='y') 
	plt.plot(x,r_hand[:,[2]], 'b', label='z') 
	plt.xlabel("{} frames".format(tot_fr)) 	
	plt.ylabel("mm") 

	plt.subplot(2, 2, 2)
	plt.title("Left hand Trajectory")
	plt.grid(True)

	plt.plot(x,l_hand[:, [0]], 'r') 
	plt.plot(x,l_hand[:, [1]], 'g') 
	plt.plot(x,l_hand[:, [2]], 'b') 
	plt.xlabel("{} frames".format(tot_fr)) 	
	plt.ylabel("mm") 

	plt.subplot(2, 2, 3)
	plt.title("RWRE Trajectory")
	plt.grid(True)

	plt.plot(x,r_arm[:,[0]], 'r') 
	plt.plot(x,r_arm[:,[1]], 'g') 
	plt.plot(x,r_arm[:,[2]], 'b') 
	plt.xlabel("{} frames".format(tot_fr)) 	
	plt.ylabel("movement") 

	plt.subplot(2, 2, 4)
	plt.title("LWRE Trajectory")
	plt.grid(True)

	plt.plot(x,l_arm[:,[0]], 'r') 
	plt.plot(x,l_arm[:,[1]], 'g') 
	plt.plot(x,l_arm[:,[2]], 'b') 
	plt.xlabel("{} frames".format(tot_fr)) 	
	plt.ylabel("mm") 

	legend = fig.legend(loc='upper right')
	plt.show()

def is_rest_pose(frame, data, mlist):
	# artificial marker in the middle of the hand for all frames 
	r_mid_hand = hand_marker(data, mlist, 'R')[frame,:]
	l_mid_hand = hand_marker(data, mlist, 'L')[frame,:]

	#position of the guidlines
	STRN = data[frame, marker_index(mlist, 'STRN'), :]
	RFSH = data[frame, marker_index(mlist, 'RFSH'), :]
	LFSH = data[frame, marker_index(mlist, 'LFSH'), :]
	RFWT = data[frame, marker_index(mlist, 'RFWT'), :]
	LFWT = data[frame, marker_index(mlist, 'LFWT'), :]

	vicinity_v = abs((RFSH[2]-RFWT[2]))*0.3 # by vertical
	vicinity = abs((RFSH[0]-LFWT[0]))*0.3 # by horizontal
	is_rp = 0
	if(r_mid_hand[2] <= RFWT[2]+vicinity_v and l_mid_hand[2] <= LFWT[2]+vicinity_v):
		if(r_mid_hand[0] > RFWT[0]-vicinity and r_mid_hand[0] < LFWT[0]+vicinity):
			if(l_mid_hand[0] > RFWT[0]-vicinity and l_mid_hand[0] < LFWT[0]+vicinity):
				# if(l_mid_hand[0])
				is_rp = 1

	return is_rp		

def hand_location_in_frame(frame, data, mlist, h='R'):
	location = 0

	# artificial marker in the middle of the hand for all frames 
	mid_hand = hand_marker(data, mlist, h)[frame,:]
	
	#position of the guidlines
	ARIEL = data[frame, marker_index(mlist, 'ARIEL'), :]
	RFSH = data[frame, marker_index(mlist, 'RFSH'), :]
	LFSH = data[frame, marker_index(mlist, 'LFSH'), :]
	CLAV = data[frame, marker_index(mlist, 'CLAV'), :]
	STRN = data[frame, marker_index(mlist, 'STRN'), :]
	C7 = data[frame, marker_index(mlist, 'C7'), :]
	RFWT = data[frame, marker_index(mlist, 'RFWT'), :]
	LFWT = data[frame, marker_index(mlist, 'LFWT'), :]

	vicinity_v = abs((RFSH[2]-RFWT[2]))*0.05 # by vertical
	vicinity = abs((RFSH[0]-LFSH[0]))*0.1 # by horizontal
	# print(vicinity)
	# print(vicinity_v)

	if(mid_hand[2] >= ARIEL[2] - vicinity_v ):											# if above head
		if(mid_hand[0] < RFSH[0]-vicinity or mid_hand[0] < RFWT[0]-vicinity):
			location = 1
		elif(mid_hand[0] <= LFSH[0]+vicinity or mid_hand[0] <= RFWT[0]+vicinity):
			location = 2
		else: #if(mid_hand > LFSH[0] or mid_hand > LFWT[0]):
			location = 3
	elif(mid_hand[2] >= RFSH[2] - vicinity_v or mid_hand[2] >= LFSH[2] - vicinity_v):	# if in front of the head
		if(mid_hand[0] < RFSH[0]-vicinity or mid_hand[0] < RFWT[0]-vicinity):
			location = 4
		elif(mid_hand[0] <= LFSH[0]+vicinity or mid_hand[0] <= RFWT[0]+vicinity):
			location = 5
		else: #if(mid_hand > LFSH[0] or mid_hand > LFWT[0]):
			location = 6
	elif(mid_hand[2] >= STRN[2] - vicinity_v ):											# if in front of upper part of tourso
		if(mid_hand[0] < RFSH[0]-vicinity or mid_hand[0] < RFWT[0]-vicinity):
			location = 7
		elif(mid_hand[0] <= LFSH[0]+vicinity or mid_hand[0] <= RFWT[0]+vicinity):
			location = 8
		else: #if(mid_hand > LFSH[0] or mid_hand > LFWT[0]):
			location = 9
	elif(mid_hand[2] >= RFWT[2] - vicinity_v or mid_hand[2] >= LFWT[2] - vicinity_v):	# if in front of down part of tourso
		if(mid_hand[0] < RFSH[0]-vicinity or mid_hand[0] < RFWT[0]-vicinity):
			location = 10
		elif(mid_hand[0] <= LFSH[0]+vicinity or mid_hand[0] <= RFWT[0]+vicinity):
			location = 11
		else: #(mid_hand > LFSH[0] or mid_hand > LFWT[0]):
			location = 12
	elif(mid_hand[2] < RFWT[2]-vicinity_v or mid_hand[2] < LFWT[2]-vicinity_v):			# if below the weist
		if(mid_hand[0] < RFSH[0] + vicinity or mid_hand[0] < RFWT[0] + vicinity):
			location = 13
		elif(mid_hand[0] <= LFSH[0] + vicinity or mid_hand[0] <= LFWT[0] + vicinity):
			location = 14
		elif(mid_hand[0] > LFSH[0] - vicinity or mid_hand[0] > LFWT[0] - vicinity):
			location = 15

	return location

def hand_location(start_frame, end_frame, data, mlist, h='R'):
	tot_frames = end_frame - start_frame
	temp_data = data[start_frame:end_frame+1, :, :]
	hand_locations = np.zeros([tot_frames, 2], dtype = int)
	change_counter = 0

	rest_pose = []

	for frame in range(0, tot_frames, 1):
		hand_locations[frame] = (frame+start_frame, hand_location_in_frame(frame, temp_data, mlist, h))
		if(hand_locations[frame][1] != hand_locations[frame-1][1]):
			change_counter = change_counter+1
		if(h=='R'):
			if(is_rest_pose(frame, temp_data, mlist) == 1):
				rest_pose.append(frame)

	# print(hand_locations)
	# print(rest_pose)
	rest_pose = np.array(rest_pose)
	return hand_locations, change_counter, rest_pose

def plot_hands_location(start_frame, end_frame, data, mlist):
	x = np.arange(start_frame+1,end_frame+1 )
	y1, c1, rp = hand_location(start_frame, end_frame, data, mlist, h='R') 
	y2, c2, r = hand_location(start_frame, end_frame, data, mlist, h='L') 

	# print("R-hand location changes: {}".format(c1))
	# print("L-hand location changes: {}".format(c2))

	fig = plt.figure("HandsLocation", figsize=(10.5,7))
	fig.suptitle("Hands location during sign between {} and {} frame".format(start_frame, end_frame))
	
	# plt.subplot(2, 1, 1)
	plt.plot(x,y1[:,[1]], 'r', label='Right hand') 
	plt.plot(x,y2[:,[1]], 'g', label='Left hand') 
	plt.plot(x[rp],y1[:,[1]][rp], 'bo', label='rest pose') 
	plt.grid(True)
	plt.xlabel("Frames")	
	plt.ylabel("Regions")

	# plt.subplot(2, 2, 3)
	# plt.title("R-hand location changes: {}".format(c1))
	# plt.hist(y1[:,[1]], bins=range(15), facecolor='r', align="left")
	# plt.xticks(np.arange(1, 16, step=1)) 
	# plt.ylabel("Number of Frames")	
	# plt.xlabel("Regions")

	# plt.subplot(2, 2, 4)
	# plt.title("L-hand location changes: {}".format(c2))
	# plt.hist(y2[:,[1]], bins=range(15),  facecolor='g', align="left") 
	# plt.xticks(np.arange(1, 16, step=1)) 
	# plt.ylabel("Number of Frames")	
	# plt.xlabel("Regions")
	
	legend = fig.legend(loc='upper right')
	plt.show()

def hand_speed(start_frame, end_frame, data, mlist, h='R'):
	tot_frames = end_frame - start_frame
	temp_data = data[start_frame:end_frame+1, :, :]
	speed = np.zeros([tot_frames, 3], dtype = float)
	speed_change = np.zeros([tot_frames, 3], dtype = float)

	hand = hand_marker(temp_data, mlist, h)

	for frame in range(0, tot_frames,1):
		speed[frame] = hand[frame+1,:] - hand[frame,:]

	for frame in range(0, tot_frames-1,1):
		speed_change[frame] =(speed[frame+1]-speed[frame])

	return speed, speed_change

def hand_speed2(start_frame, end_frame, data, mlist, h='R'):
	tot_frames = end_frame - start_frame
	# temp_data = data[start_frame:end_frame+1, :, :]
	hand, arm = hand_trajectory(start_frame, end_frame, data, mlist, h)
	hand_n, arm = norm_trajectory(hand, arm)
	speed = np.zeros([tot_frames, 1], dtype = float)
	speed_change = np.zeros([tot_frames, 1], dtype = float)

	# hand = hand_marker(temp_data, mlist, h)

	for frame in range(0, tot_frames-1,1):
		speed[frame] = hand_n[frame+1] - hand_n[frame]

	for frame in range(0, tot_frames-1,1):
		speed_change[frame] =(speed[frame+1]-speed[frame])

	return speed, speed_change

def hand_acc(velocity):
	n = len(velocity)
	acc = np.zeros([n, 1], dtype=float)

	for frame in range(0, n-1,1):
		acc[frame] = velocity[frame+1]-velocity[frame]

	return acc
# normilize the hand speed and acceleration for the 3 axises
def norm_speed(velocity, acceleration):
	v = np.linalg.norm(velocity, axis=1)
	a = np.linalg.norm(acceleration, axis=1)

	return(v, a)

#not finished
def wrist_orientation_frame(frame, data, mlist, h='R'):
	if(h == 'L'):
		LOWR = data[frame, marker_index(mlist, 'LOWR'), :]
		LIWR = data[frame, marker_index(mlist, 'LIWR'), :]
		wr_vector = (LIWR - LOWR)

		LOHAND = data[frame, marker_index(mlist, 'LOHAND'), :]
		LIHAND = data[frame, marker_index(mlist, 'LIHAND'), :]
		mid_hand = (LOHAND + LIHAND)/2
		mid_wr = (LOWR + LIWR)/2
		palm_vector = ( mid_hand - mid_wr)

	else:
		ROWR = data[frame, marker_index(mlist, 'ROWR'), :]
		RIWR = data[frame, marker_index(mlist, 'RIWR'), :]
		wr_vector = (RIWR - ROWR)

		ROHAND = data[frame, marker_index(mlist, 'ROHAND'), :]
		RIHAND = data[frame, marker_index(mlist, 'RIHAND'), :]
		mid_hand = (ROHAND + RIHAND)/2
		mid_wr = (ROWR + RIWR)/2
		palm_vector = (mid_hand - mid_wr )

	# print(wr_vector)
	# print(palm_vector)
	print(vm.angle(wr_vector, palm_vector))

#not finished
def finger_orientation_frame(frame, data, mlist, h='R'):
	if(h == 'L'):
		thb0_v = data[frame, marker_index(mlist, 'LTHB1'), :] - data[frame, marker_index(mlist, 'LTHB0'), :]
		thb1_v = data[frame, marker_index(mlist, 'LTHB2'), :] - data[frame, marker_index(mlist, 'LTHB1'), :]

		idx0_v = data[frame, marker_index(mlist, 'LIDX2'), :] - data[frame, marker_index(mlist, 'LIDX1'), :]
		idx1_v = data[frame, marker_index(mlist, 'LIDX3'), :] - data[frame, marker_index(mlist, 'LIDX2'), :]

		md0_v = data[frame, marker_index(mlist, 'LMD2'), :] - data[frame, marker_index(mlist, 'LMD1'), :]
		md1_v = data[frame, marker_index(mlist, 'LMD3'), :] - data[frame, marker_index(mlist, 'LMD2'), :]

		rng0_v = data[frame, marker_index(mlist, 'LRNG2'), :] - data[frame, marker_index(mlist, 'LRNG1'), :]
		rng1_v = data[frame, marker_index(mlist, 'LRNG3'), :] - data[frame, marker_index(mlist, 'LRNG2'), :]

		ltl0_v = data[frame, marker_index(mlist, 'LTL2'), :] - data[frame, marker_index(mlist, 'LTL1'), :]
		ltl1_v = data[frame, marker_index(mlist, 'LTL3'), :] - data[frame, marker_index(mlist, 'LTL2'), :]

		LOWR = data[frame, marker_index(mlist, 'LOWR'), :]
		LIWR = data[frame, marker_index(mlist, 'LIWR'), :]
		# wr_vector = (LIWR - LOWR)

		LOHAND = data[frame, marker_index(mlist, 'LOHAND'), :]
		LIHAND = data[frame, marker_index(mlist, 'LIHAND'), :]
		mid_hand = (LOHAND + LIHAND)/2
		mid_wr = (LOWR + LIWR)/2
		palm_vector = ( mid_hand - mid_wr)

	else:
		thb0_v = data[frame, marker_index(mlist, 'RTHB1'), :] - data[frame, marker_index(mlist, 'RTHB0'), :]
		thb1_v = data[frame, marker_index(mlist, 'RTHB2'), :] - data[frame, marker_index(mlist, 'RTHB1'), :]

		idx0_v = data[frame, marker_index(mlist, 'RIDX2'), :] - data[frame, marker_index(mlist, 'RIDX1'), :]
		idx1_v = data[frame, marker_index(mlist, 'RIDX3'), :] - data[frame, marker_index(mlist, 'RIDX2'), :]

		md0_v = data[frame, marker_index(mlist, 'RMD2'), :] - data[frame, marker_index(mlist, 'RMD1'), :]
		md1_v = data[frame, marker_index(mlist, 'RMD3'), :] - data[frame, marker_index(mlist, 'RMD2'), :]

		rng0_v = data[frame, marker_index(mlist, 'RRNG2'), :] - data[frame, marker_index(mlist, 'RRNG1'), :]
		rng1_v = data[frame, marker_index(mlist, 'RRNG3'), :] - data[frame, marker_index(mlist, 'RRNG2'), :]

		ltl0_v = data[frame, marker_index(mlist, 'RTL2'), :] - data[frame, marker_index(mlist, 'RTL1'), :]
		ltl1_v = data[frame, marker_index(mlist, 'RTL3'), :] - data[frame, marker_index(mlist, 'RTL2'), :]

		ROWR = data[frame, marker_index(mlist, 'ROWR'), :]
		RIWR = data[frame, marker_index(mlist, 'RIWR'), :]
		# wr_vector = (ROWR - RIWR)

		ROHAND = data[frame, marker_index(mlist, 'ROHAND'), :]
		RIHAND = data[frame, marker_index(mlist, 'RIHAND'), :]
		mid_hand = (ROHAND + RIHAND)/2
		mid_wr = (ROWR + RIWR)/2
		palm_vector = (mid_hand - mid_wr )

#not finished
def palm_orientation_frame(frame, data, mlist, h='R'):
	if(h == 'L'):
		LOWR = data[frame, marker_index(mlist, 'LOWR'), :]
		LIWR = data[frame, marker_index(mlist, 'LIWR'), :]
		
		LOHAND = data[frame, marker_index(mlist, 'LOHAND'), :]
		LIHAND = data[frame, marker_index(mlist, 'LIHAND'), :]
		
		p = LOHAND - LIHAND
		q = LOWR - LIHAND

	else:
		ROWR = data[frame, marker_index(mlist, 'ROWR'), :]
		# print("ROWR: {}".format(ROWR))
		RIWR = data[frame, marker_index(mlist, 'RIWR'), :]
		
		ROHAND = data[frame, marker_index(mlist, 'ROHAND'), :]
		RIHAND = data[frame, marker_index(mlist, 'RIHAND'), :]
		
		p = ROHAND - RIHAND
		q = ROWR - RIHAND
	
	palm_plane = vm.crossproduct(p,q)

	RFWT = data[frame, marker_index(mlist, 'RFWT'), :]
	LFWT = data[frame, marker_index(mlist, 'LFWT'), :]
	STRN = data[frame, marker_index(mlist, 'STRN'), :]
	RBWT = data[frame, marker_index(mlist, 'RBWT'), :]

	a = RFWT - LFWT
	b = STRN - LFWT
	c = RBWT - LFWT

	body_plane = vm.crossproduct(a,b)
	floor_plane = vm.crossproduct(a,c)

	print("palm_plane={}".format(palm_plane))
	print("body_plane={}".format(body_plane))
	print("floor_plane={}".format(floor_plane))

	print("< palm, body: {} ".format(vm.angle(palm_plane,body_plane)))
	print("< palm, floor: {} ".format(vm.angle(palm_plane,floor_plane)))

def plot_hand1(start_frame, end_frame, data, mlist, fps):
	r_hand_tr, r_arm_tr = hand_trajectory(start_frame, end_frame, data, mlist, 'R')
	r_h_tr, r_a_tr = norm_trajectory(r_hand_tr, r_arm_tr)
	l_hand_tr, l_arm_tr = hand_trajectory(start_frame, end_frame, data, mlist, 'L')
	l_h_tr, l_a_tr = norm_trajectory(l_hand_tr, l_arm_tr)

	r_velocity, r_acceleration = hand_speed(start_frame, end_frame, data, mlist, 'R')
	r_vel, r_acc = norm_speed(r_velocity, r_acceleration)
	l_velocity, l_acceleration = hand_speed(start_frame,end_frame, data, mlist, 'L' )
	l_vel, l_acc = norm_speed(l_velocity, l_acceleration)

	r_loc, r_ch = hand_location(start_frame, end_frame, data, mlist, 'R')
	l_loc, r_ch = hand_location(start_frame, end_frame, data, mlist, 'L')	
	
	x = np.arange(start_frame, end_frame)

	plt.subplot
	fig = plt.figure("RightHand", figsize=(10.5,7))
	fig.suptitle("Right hand speed and location for sign between {} and {} frame".format(start_frame, end_frame))	

	n = end_frame-start_frame
	k = np.arange(n)
	T = n/fps
	frq = k/T

	r_hand_fft = np.fft.fft(r_h_tr)/n
	r_vel_fft = np.fft.fft(r_vel)/n
	r_acc_fft = np.fft.fft(r_acc)/n
	

	plt.subplot(4, 2, 1)
	# plt.plot(x,r_hand_tr[:,[0]], 'r', label='x') 
	# plt.plot(x,r_hand_tr[:,[1]], 'g', label='y') 
	# plt.plot(x,r_hand_tr[:,[2]], 'b', label='z') 
	plt.plot(x, r_h_tr, 'r', label="Right hand")
	plt.plot(x, l_h_tr, 'g', label="Left hand")
	plt.ylabel("Position (mm)") 
	plt.grid(True)

	#freq1 = np.fft.fftfreq(r_hand_fft.shape[-1])
	

	plt.subplot(4,2, 2)
	plt.plot(frq, r_hand_fft, 'r')
	plt.ylabel("pos frequencie domain") 
	plt.grid(True)

	plt.subplot(4, 2, 3)
	# plt.plot(x,r_velocity[:,[0]], 'r') 
	# plt.plot(x,r_velocity[:,[1]], 'g') 
	# plt.plot(x,r_velocity[:,[2]], 'b') 
	plt.plot(x, r_vel, 'r')
	plt.plot(x, l_vel, 'g')
	plt.ylabel("Speed (mm/frame)") 
	plt.grid(True)
		
	plt.subplot(4,2,4)
	plt.plot(frq, r_vel_fft, 'r')
	plt.ylabel("Speed frequencie domain") 
	plt.grid(True)


	plt.subplot(4, 2, 5)
	plt.plot(x,r_acc, 'r')
	plt.plot(x,l_acc, 'g')
	plt.ylabel("Acceleration (mm/frame^2)")
	plt.grid(True)

	freq2 = np.fft.fftfreq(r_acc.shape[-1])
	plt.subplot(4, 2, 6)
	plt.plot(frq, r_acc_fft, 'r')
	plt.ylabel("Acceleration ")
	plt.grid(True)


	plt.subplot(4, 1, 4)	
	plt.plot(x,r_loc[:,[1]], 'r') 
	plt.plot(x,l_loc[:,[1]], 'g') 
	plt.xlabel("Frames")	
	plt.ylabel("Regions")
	plt.grid(True)

	legend = fig.legend(loc='upper right')
	
	plt.show()
	# plotSpectrum(r_h_tr, 120.0)
	# plotSpectrum(r_vel, 120.0)
	# plotSpectrum(r_acc, 120.0)

# params for Buterworth filter
def butter_coef(cutoff, fs, order=5):
	nyq = 0.5*fs
	Wn = cutoff/nyq # Cutoff frequency
	b,a = signal.butter(order, Wn, output='ba') #order 
	return b,a
# apply the filter
def butter_filter(data, cutoff, fs, order=5):
	b,a = butter_coef(cutoff, fs, order)
	y = signal.filtfilt(b, a, data, axis = 0)
	return y

def plot_hand2(start_frame, end_frame, data, mlist, fps, title):
	r_hand_tr, r_arm_tr = hand_trajectory(start_frame, end_frame, data, mlist, 'R')
	r_hand_tr_norm, r_arm_tr_norm = norm_trajectory(r_hand_tr, r_arm_tr)
	l_hand_tr, l_arm_tr = hand_trajectory(start_frame, end_frame, data, mlist, 'L')
	l_hand_tr_norm, l_arm_tr_norm = norm_trajectory(l_hand_tr, l_arm_tr)

	r_velocity, r_acceleration = hand_speed(start_frame, end_frame, data, mlist, 'R')
	r_vel_norm, r_acc_norm = norm_speed(r_velocity, r_acceleration)
	l_velocity, l_acceleration = hand_speed(start_frame,end_frame, data, mlist, 'L' )
	l_vel_norm, l_acc_norm = norm_speed(l_velocity, l_acceleration)

	r_loc, r_ch, r = hand_location(start_frame, end_frame, data, mlist, 'R')
	l_loc, l_ch, l = hand_location(start_frame, end_frame, data, mlist, 'L')	
	
	x = np.arange(start_frame, end_frame)

	plt.subplot
	fig = plt.figure(title, figsize=(10.5,7))
	fig.suptitle("Right hand speed and location for sign between {} and {} frame".format(start_frame, end_frame))	

	# FFT 
	n = end_frame-start_frame
	k = np.arange(n)
	T = n/fps
	frq = k/T

	r_hand_fft = np.fft.fft(r_hand_tr_norm)//n
	r_vel_fft = np.fft.fft(r_vel_norm)/n
	r_acc_fft = np.fft.fft(r_acc_norm)/n
	
	r_hand_tr_filtered = butter_filter(r_hand_tr_norm, 12, fps, 5)
	r_hand_vel_filt = butter_filter(r_vel_norm, 12, fps, 5)
	
	r_hand_acc_filt = butter_filter(r_acc_norm, 12, fps, 5)
	r_hand_acc_filt1 = butter_filter(r_acc_norm, 12, fps, 10)
	r_hand_acc_filt2 = butter_filter(r_acc_norm, 12, fps, 12)
	# print(r_hand_tr_filtered)
	# r_velocity_f, r_acceleration_f = hand_speed(start_frame, end_frame, r_hand_tr_filtered, mlist, 'R')

	plt.subplot(4, 2, 1)
	plt.plot(x, r_hand_tr_norm, 'r-', label="Original")
	plt.ylabel("Position (mm)") 
	plt.grid(True)

	plt.subplot(4, 2, 2)
	plt.plot(frq, r_hand_fft, 'g-')
	plt.ylabel("Pos frequencie domain") 
	plt.grid(True)

	plt.subplot(4, 2, 3)
	plt.plot(x, r_vel_norm, 'r-')
	plt.ylabel("Velocity (mm/frame)") 
	plt.grid(True)
	plt.subplot(4, 2, 4)
	plt.plot(x, r_hand_vel_filt, 'g-', label="Filtered")
	plt.ylabel("Velocity (mm/frame)") 
	plt.grid(True)

	plt.subplot(5, 1, 4)
	plt.plot(x, r_acc_norm, 'r-')
	plt.plot(x, r_hand_acc_filt, 'g-')
	plt.ylabel("Acceleration (mm/frame^2)") 
	plt.grid(True)


	

	legend = fig.legend(loc='upper right')
	
	plt.show()

def find_extremums(data):
	maxs = argrelextrema(data, np.greater, order=10)
	mins = argrelextrema(data, np.less, order=10)
	extr = np.sort(np.append(maxs, mins ))

	return extr

def segm(vel, acc, start_frame, end_frame, data, mlist):
	vel_ex = zero_crossing(acc)
	n = len(vel_ex)
	threshold = np.median(vel) # 0.15


	labels = np.full([n,3], -100)
	for i in range(n):
		if(vel[vel_ex[i]] <= threshold):
			labels[i][0] = vel_ex[i]
			labels[i][1] = 0
		else: 
			labels[i][0] = vel_ex[i]
			labels[i][1] = 1

	i=1
	while(i <n-2):
		if(labels[i][1]-labels[i+1][1] < 0):
			# mark start markers
			if(labels[i-1][1] == 0 ):
				if(is_rest_pose(labels[i][0]+start_frame, data, mlist) == 1 ):
					labels[i][2] = 1
				else:
					labels[i][2] = -15
			else:
				labels[i][2] = -1
		# elif(is_rest_pose(labels[i+1][0]+start_frame, data, mlist) == 1 ):
		elif(labels[i][1]-labels[i+1][1] > 0): 
			# print("-- - probably end")
			# mark end markers
			if(labels[i+2][1] == 0):
				if(is_rest_pose(labels[i+1][0]+start_frame, data, mlist) == 1 ):
				# print("---- - probably end")
					labels[i][2] = -1
					labels[i+1][2] = 0
					i = i+1
			else:
				labels[i][2] = -1
		else:
			labels[i][2] = -5
		# print(labels[i,:])
		i = i+1

	return labels 

def segm1(acc, acc_ex, start_frame, end_frame, data, mlist, h='R'):
	
	n = len(acc_ex)
	threshold = 0.012

	# n frames  - good extremum - start 1/end 0 /not important -1
	labels = np.full([n,3], -1)
	for i in range(n):
		if(acc[acc_ex[i]] <= threshold):
			labels[i][0] = acc_ex[i]
			labels[i][1] = 0
		else: 
			labels[i][0] = acc_ex[i]
			labels[i][1] = 1
	
	i=1
	labels[0][2] = -1
	while(i <n-2):
		if(is_rest_pose(labels[i][0]+start_frame, data, mlist) ==1 ):
			if(labels[i][1]-labels[i+1][1] < 0):
				# mark start markers
				if(labels[i-1][1] == 0 and labels[i+2][1] == 1):
					labels[i][2] = 1
				else:
					labels[i][2] = -1
			elif(labels[i][1]-labels[i+1][1] > 0): 
				# print("-- - probably end")
				# mark end markers
				if(labels[i+2][1] == 0):
					# print("---- - probably end")
					labels[i][2] = -1
					labels[i+1][2] = 0
					i = i+1
				else:
					labels[i][2] = -1
			else:
				labels[i][2] = -1
		# print(labels[i,:])
		i = i+1
	# print(labels)
	return labels

def get_signs_borders(labels, start_frame, end_frame):
	start=0
	end=0
	#labels[i][0,1,2]
	# n frames  - good extremum - start 1/end 0 /not important -1
	i = 0 
	count = 0 
	signs = []
	n = len(labels)
	signs2 = []

	while(i < n-1):
		if(labels[i][2] == 1 ):
			start = i
			i=i+1
			s = i
			while(i<n-1):
				if(labels[i][2] == 0 ):
					break
				if(labels[i][2] == 1 ):
					i = i - 2 
					break
				i = i +1
			end = i
			e = i - 2
			# print(labels[end][0]+start_frame)
			count = count + 1
			signs.append((labels[start][0], labels[end][0]))
			signs2.append((labels[s][0], labels[e][0]))
			print("sign #{}: {}     -     {} ". format(count,labels[start][0]+start_frame, labels[end][0]+start_frame))
			print("sign #{}:     {} - {} ". format(count,labels[s][0]+start_frame, labels[e][0]+start_frame))
			print()
		i = i +1

	signs = np.array(signs)
	signs2 = np.array(signs2)
	return signs, signs2

def segment_signs(start_frame, end_frame, data, mlist, fps):
	h = 'R'
	if(dominant_hand(start_frame, end_frame, data, mlist) == 1):
		h = 'R'
	elif(dominant_hand(start_frame, end_frame, data, mlist) == 2):
		h = 'L'

	velocity, acceleration = hand_speed(start_frame, end_frame, data, mlist, h)
	vel_norm, acc_norm = norm_speed(velocity, acceleration)
	
	acc = butter_filter(acc_norm, 5, fps, 10)
	acc_ex = find_extremums(acc)

	labels = segm(acc, acc_ex, start_frame, end_frame, data, mlist, h)
	signs = get_signs_borders(labels, start_frame, end_frame)

	r_loc, r_ch, rp = hand_location(start_frame, end_frame, data, mlist, 'R')
	l_loc, r_ch, r = hand_location(start_frame, end_frame, data, mlist, 'L')

	count = len(signs)
	x = np.arange(start_frame, end_frame)
	
	fig = plt.figure("Signs", figsize=(10.5,7))
	plt.suptitle("{} signs between {} and {} frame".format(count, start_frame, end_frame))

	plt.subplot(2, 1, 1)
	plt.plot(x, acc_norm, 'y-', label='Normalized acceleration')
	plt.plot(x, acc, 'r-', label='Filtered acceleration')
	plt.plot(x[acc_ex], acc[acc_ex], 'r*', label='Extremums')	
	if(len(signs)>0):
		plt.plot(x[signs[:, 0]], acc[signs[:, 0]], 'bs', label = "Start")	
		plt.plot(x[signs[:,1]], acc[signs[:,1]], 'o', label = "End")	
	plt.ylabel("Acceleration (mm/frame^2)") 
	plt.grid(True)


	plt.subplot(2, 1, 2)	
	plt.plot(x,r_loc[:,[1]], 'r', label="Right hand") 
	plt.plot(x,l_loc[:,[1]], 'g', label="Left hand") 
	plt.plot(x[rp], r_loc[:,[1]][rp], 'yo', label="Rest pose") 
	plt.xlabel("Frames")	
	plt.ylabel("Regions")
	plt.grid(True)

	legend = fig.legend(loc='upper right')

	# plot_hands_location(start_frame, end_frame, data, mlist)
	
	plt.show()

	return signs

def get_segmented_data(start_frame, end_frame, data, mlist, fps):
	signs_boundaries = segment_signs(start_frame, end_frame, data, mlist, fps)
	count = len(signs_boundaries)
	s_d = np.zeros((int(data.shape[0]), int(data.shape[1]), int(data.shape[2])) )
	# # print(data.shape)
	# # print(s_d.shape)
	# s_d = []
	j = 0 
	n_sum = 0 
	for i in range(0,count):
		st = signs_boundaries[i][0]+start_frame
		end = signs_boundaries[i][1]+start_frame
		n = end-st
		n_sum = n_sum + n
		# print(n)
		# s_d.append(data[st:end, :, :])
		s_d[j:j+n, :, :] = data[st:end, :, :]
		j = j + n
	# s_d = np.array(s_d)
	# print(np.shape(s_d))
	s_d = np.delete(s_d, np.s_[n_sum:], 0)
	# print(np.shape(s_d))
	return s_d

def plot_segmentation(start_frame, end_frame, data, mlist, fps) :
	
	velocity, acceleration = hand_speed(start_frame, end_frame, data, mlist, 'R')
	vel_norm, acc_norm = norm_speed(velocity, acceleration)
	acc = butter_filter(acc_norm, 5, fps, 10)
	acc_ex = find_extremums(acc)

	x = np.arange(start_frame, end_frame)
	signs  = segment_signs(start_frame, end_frame, data, mlist, fps)

	r_loc, r_ch, rp = hand_location(start_frame, end_frame, data, mlist, 'R')
	l_loc, r_ch, r = hand_location(start_frame, end_frame, data, mlist, 'L')	

	plt.subplot
	fig = plt.figure("Signs", figsize=(10.5,7))

	plt.subplot(2, 1, 1)
	plt.plot(x, acc_norm, 'y-')
	plt.plot(x, acc, 'r-')
	plt.plot(x[acc_ex], acc[acc_ex], 'r*')	
	if(len(signs)>0):
		plt.plot(x[signs[:, 0]], acc[signs[:, 0]], 'bs', label = "Start")	
		plt.plot(x[signs[:,1]], acc[signs[:,1]], 'o', label = "End")	
	plt.ylabel("Acceleration (mm/frame^2)") 
	plt.grid(True)

	plt.subplot(2, 1, 2)	
	plt.plot(x,r_loc[:,[1]], 'r') 
	plt.plot(x,l_loc[:,[1]], 'g') 
	plt.plot(x[rp], r_loc[:,[1]][rp], 'go', label="Rest pose") 
	plt.xlabel("Frames")	
	plt.ylabel("Regions")
	plt.grid(True)

	legend = fig.legend(loc='upper right')
	
	plt.show()

def plot_hand(start_frame, end_frame, data, mlist, fps):
	r_hand_tr, r_arm_tr = hand_trajectory(start_frame, end_frame, data, mlist, 'R')
	r_hand_tr_norm, r_arm_tr_norm = norm_trajectory(r_hand_tr, r_arm_tr)

	r_tr_extr = find_extremums(r_hand_tr_norm)

	l_hand_tr, l_arm_tr = hand_trajectory(start_frame, end_frame, data, mlist, 'L')
	l_hand_tr_norm, l_arm_tr_norm = norm_trajectory(l_hand_tr, l_arm_tr)
	l_tr_extr = find_extremums(l_hand_tr_norm)
	
	r_velocity, r_acceleration = hand_speed(start_frame, end_frame, data, mlist, 'R')
	r_vel_norm, r_acc_norm = norm_speed(r_velocity, r_acceleration)
	r_vel_extr = find_extremums(r_vel_norm)

	l_velocity, l_acceleration = hand_speed(start_frame,end_frame, data, mlist, 'L' )
	l_vel_norm, l_acc_norm = norm_speed(l_velocity, l_acceleration)
	l_vel_extr = find_extremums(l_vel_norm)

	r_loc, r_ch, rp = hand_location(start_frame, end_frame, data, mlist, 'R')
	l_loc, r_ch, r = hand_location(start_frame, end_frame, data, mlist, 'L')	
	
	x = np.arange(start_frame, end_frame)

	# data, cutoff frequency, fps, order # for wrist is around 2
	r_hand_acc_filt = butter_filter(r_acc_norm, 5, fps, 10)
	r_acc_extr = find_extremums(r_hand_acc_filt)

	l_hand_acc_filt = butter_filter(l_acc_norm, 5, fps, 10)
	l_acc_extr = find_extremums(l_hand_acc_filt)
	
	# signs  = segment_signs(start_frame, end_frame, data, mlist, fps)

	
	plt.subplot
	fig = plt.figure("Hand", figsize=(10.5,7))
	fig.suptitle("Hand speed and location for sign between {} and {} frame".format(start_frame, end_frame))	

	plt.subplot(4, 2, 1)
	plt.plot(x, r_hand_tr_norm, 'r-', label="Right hand")
	plt.plot(x[r_tr_extr], r_hand_tr_norm[r_tr_extr], 'r*', label="Extremum")
	plt.ylabel("Position (mm)") 
	plt.grid(True)


	plt.subplot(4, 2, 2)
	plt.plot(x, l_hand_tr_norm, 'g-', label="Left hand")
	plt.plot(x[l_tr_extr], l_hand_tr_norm[l_tr_extr], 'g*', label="Extremum")
	plt.ylabel("Position (mm)") 
	plt.grid(True)

	plt.subplot(4, 2, 3)
	plt.plot(x, r_vel_norm, 'r-')
	plt.plot(x[r_vel_extr], r_vel_norm[r_vel_extr], 'r*')	
	plt.ylabel("Velocity (mm/frame)") 
	plt.grid(True)
	
	plt.subplot(4, 2, 4)
	plt.plot(x, l_vel_norm, 'g-')
	plt.plot(x[l_vel_extr], l_vel_norm[l_vel_extr], 'g*')
	plt.ylabel("Velocity (mm/frame)") 
	plt.grid(True)

	plt.subplot(4, 2, 5)
	plt.plot(x, r_acc_norm, 'y-')
	plt.plot(x, r_hand_acc_filt, 'r-')
	plt.plot(x[r_acc_extr], r_hand_acc_filt[r_acc_extr], 'r*')		
	plt.ylabel("Acceleration (mm/frame^2)") 
	plt.grid(True)

	plt.subplot(4, 2, 6)
	plt.plot(x, l_acc_norm, 'b-')
	plt.plot(x, l_hand_acc_filt, 'g-')
	plt.plot(x[l_acc_extr], l_hand_acc_filt[l_acc_extr], 'g*')
	plt.ylabel("Acceleration (mm/frame^2)") 
	plt.grid(True)

	plt.subplot(4, 1, 4)	
	plt.plot(x,r_loc[:,[1]], 'r') 
	plt.plot(x,l_loc[:,[1]], 'g') 
	plt.xlabel("Frames")	
	plt.ylabel("Regions")
	plt.grid(True)

	legend = fig.legend(loc='upper right')
	
	plt.show()

def plotSpectrum(y,Fs):
	"""
	Plots a Single-Sided Amplitude Spectrum of y(t)
	"""
	n = len(y) # length of the signal
	k = np.arange(n)
	T = n/Fs
	frq = k/T # two sides frequency range
	#frq = frq[range(n/2.0)] # one side frequency range

	Y = np.fft.fft(y)/n # fft computing and normalization
	#Y = Y[range(n/2)]

	plt.plot(frq,abs(Y),'r') # plotting the spectrum
	plt.xlabel('Freq (Hz)')
	plt.ylabel('|Y(freq)|')
	plt.show()
