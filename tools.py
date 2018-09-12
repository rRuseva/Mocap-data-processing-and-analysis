import numpy as np
from numpy import linalg

import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import sys
import c3d
sys.path.insert(0, "D:\Radi\Radi RU\4ti kurs\2sm-WBU\MOCAP\Python\mocap")
import vector_math as vm
from scipy.signal import argrelextrema
import scipy.signal as signal



# returns the index of a given marker from given marker list
def marker_index(mlist, mname):
	for i, m in enumerate(mlist):
		if(m == mname):
			return i
	
	return False

#returns the marker name for the given index from given list of markers
def marker_name(mlist, index):
	if(index < len(mlist)):
		return mlist[index]
	else: 
		return False

# read given c3d file; 
# returns numpy array data[frames, markers, 3 axes], list with existing markers, frame rate
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
			data[frame, tot_markers,:] = np.zeros(5)
		
		makrer_list = []
		for marker in enumerate(reader.point_labels):
			makrer_list.append(marker[1].strip())
			#print(marker)
		makrer_list.append("ORIGIN")
		tot_markers = tot_markers+1

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

# returns the 3 coordinates for all frames, of artificial marker in the middle of a hand 
def hand_marker(data, mlist, h='R'):
	# artificial marker for 
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
	# hand =(OHAND + IHAND + IWR + OWR )/4.0

	return hand

# returns 4numpy arrays for left and right hand and arm displacment
# returns right_dominant = 1 if the dominant hand is the dominant in the sign
# returns a message if the sign is one or two handed 
def hand_displacment(start_frame, end_frame, data, mlist):
	tot_frames = end_frame - start_frame
	# extract the specified data fromstart to end frame
	temp_data = data[start_frame:end_frame+1, :, :]
	
	is_right_handed = False
	is_left_handed = False

	#markers of interest
	# the centerof the hand
	r_hand = hand_marker(temp_data, mlist, 'R')
	l_hand = hand_marker(temp_data, mlist, 'L')
	
	# arm
	RWRE_indx = marker_index(mlist, 'RWRE')			
	LWRE_indx = marker_index(mlist, 'LWRE')

	# right hand displacment
	diff_right_hand = np.zeros([tot_frames, 3])
	# leght of right hand trajectory // = total distance 
	diff_right_hand_sum = 0	
	diff_RWRE = np.zeros([tot_frames, 3])
	diff_RWRE_sum = 0

	diff_left_hand = np.zeros([tot_frames, 3])
	diff_left_hand_sum = 0 
	diff_LWRE = np.zeros([tot_frames, 3])
	diff_LWRE_sum = 0

	
	for frame in range(1, tot_frames, 1):
		diff_right_hand[frame] = abs(r_hand[frame] - r_hand[frame-1])
		diff_right_hand_sum = diff_right_hand_sum + diff_right_hand[frame]
	
		diff_left_hand[frame] = abs(l_hand[frame] - l_hand[frame-1])
		diff_left_hand_sum = diff_left_hand_sum + diff_left_hand[frame]

		diff_RWRE[frame] = abs(temp_data[frame, RWRE_indx, :] - temp_data[frame-1, RWRE_indx, :])
		diff_RWRE_sum = diff_RWRE_sum + diff_RWRE[frame]
		
		diff_LWRE[frame] = abs(temp_data[frame, LWRE_indx, :] - temp_data[frame-1, LWRE_indx, :])
		diff_LWRE_sum = diff_LWRE_sum + diff_LWRE[frame]

	#avarage based on segment lenght
	diff_right_hand_sum = diff_right_hand_sum/tot_frames
	diff_left_hand_sum = diff_left_hand_sum/tot_frames
	diff_RWRE_sum = diff_RWRE_sum/tot_frames
	diff_LWRE_sum = diff_LWRE_sum/tot_frames
	
	# threshold
	differ = ((diff_right_hand_sum + diff_left_hand_sum + diff_RWRE_sum + diff_LWRE_sum)/4)*9/10
	
	# check if hand displacment is more than a threshold and thus desides if it is one or two handed sign
	if((abs(diff_right_hand_sum) > differ).any() or (abs(diff_RWRE_sum) > differ).any()):
		is_right_handed = True
	if((abs(diff_left_hand_sum) > differ).any() or (abs(diff_LWRE_sum) > differ).any()):
		is_left_handed = True


	one_hand = 0 
	if(is_left_handed  and is_right_handed ):
		# two handed sign
		one_hand = 3
	elif(is_right_handed ):
		# right handed sign
		one_hand = 1
		right_dominant = 1
	elif(is_left_handed == 1):
		# left handed sign
		one_hand = 2
		right_dominant = 0
	else:
		# no hands or error
		one_hand = 0

	# checks if the right hand is more active than the left hand
	if((diff_right_hand_sum > diff_left_hand_sum).any() or (diff_RWRE_sum > diff_LWRE_sum).any()):
		right_dominant = 1
	else:
		right_dominant = 0

	return diff_right_hand, diff_left_hand, diff_RWRE, diff_LWRE, is_right_handed, one_hand

# return 1 if RIGHT hand is the dominant hand 
def dominant_hand(start_frame, end_frame, data, mlist):
	diff_right_hand, diff_left_hand, diff_RWRE, diff_LWRE, right_dominant, one_hand = hand_displacment(start_frame, end_frame, data, mlist)
	
	return right_dominant

# return number from 0 to 3 if the sign is:
# with no hands - only left hand - only right hand - both hands
def is_on_handed(start_frame, end_frame, data, mlist):
	diff_right_hand, diff_left_hand, diff_RWRE, diff_LWRE, right_dominant, one_hand = hand_displacment(start_frame, end_frame, data, mlist)
	
	return one_hand

# extracts the information for hand (one marker in the middle of the hand) 
# during the specified range of frames
def hand_trajectory(start_frame, end_frame, data, mlist, h='R'):
	temp_data = data[start_frame:end_frame, :, :]

	hand_pos = hand_marker(temp_data, mlist, h)
	arm_pos = np.zeros([end_frame-start_frame, 3])
	
	if(h == 'L'):
		arm_pos = temp_data[:, marker_index(mlist, 'LWRE'), :]
	else:
		arm_pos  = temp_data[:, marker_index(mlist, 'RWRE'), :]	

	return hand_pos, arm_pos

# normilize the hand and arm trajectory by the 3 axises
def norm_trajectory(hand_pos, arm_pos):
	hand = np.linalg.norm(hand_pos, axis=1)
	arm = np.linalg.norm(arm_pos, axis=1)

	return(hand, arm)

# return an numpy array with zero-crossing points for a given 1d array
def zero_crossing(data):
	zero_crossing = []
	# rejected = []
	n = len(data)

	for i in range(0, n-1):
		if((data[i]>=0.00 and data[i+1]<=0.00) or (data[i]<=0.00 and data[i+1]>=0.00)):
			if(abs(data[i]) < abs(data[i+1])):
				zero_crossing.append(i)
				# rejected.append(abs(data[i+1]))
			else:
				zero_crossing.append(i+1)
				# rejected.append(abs(data[i]))

	zero_crossing = np.array(zero_crossing)
	return zero_crossing

def zero_crossing1(data):
	zero_crossing = []
	rejected = []
	n = len(data)

	for i in range(0, n-1):
		if((data[i]>=0.00 and data[i+1]<=0.00) or (data[i]<=0.00 and data[i+1]>=0.00)):
			if(abs(data[i]) < abs(data[i+1])):
				zero_crossing.append(i)
				rejected.append(abs(data[i+1]))
			else:
				zero_crossing.append(i+1)
				rejected.append(abs(data[i]))

	thr = max(rejected)
	zero_crossing = np.array(zero_crossing)
	return zero_crossing, thr

def interesting_points(data):
	zero, thr = zero_crossing1(data)
	# print(len(zero))
	# print(zero)
	d1 = hand_acceleration(data)
	zero1, tr = zero_crossing1(d1)
	n = len(zero1)
	for i in range(0,n-1):
		if(data[zero1[i]]>= -thr and data[zero1[i]]<= thr ):
			zero = np.append(zero,zero1[i] )

	zero = np.sort(zero)
	# print("all points")
	# # print(zero)
	# print(len(zero))
	# print(zero)
	return zero

# check if the hand are in rest pose based on their location
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

# returns - an array with all frames in wich hands are in rest pose
def rest_pose(start_frame, end_frame, data, mlist):
	tot_frames = end_frame - start_frame
	temp_data = data[start_frame:end_frame+1, :, :]
	rest_pose = []

	for frame in range(0, tot_frames, 1):	
		if(is_rest_pose(frame, temp_data, mlist) == 1):
			rest_pose.append(frame)

	rest_pose = np.array(rest_pose)

	return rest_pose

# returns a number between 1 and 15 depending of 
# horizontal and vertical location of the hand in the frame 
# according to the body plane
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

	vicinity_v = abs((RFSH[2]-RFWT[2]))*0.1 # by vertical
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
	elif(mid_hand[2] >= STRN[2] - 2*vicinity_v ):											# if in front of upper part of tourso
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

# returns - an array with hand location for each frame 
# 		  - number of location changes
def hand_location(start_frame, end_frame, data, mlist, h='R'):
	tot_frames = end_frame - start_frame
	temp_data = data[start_frame:end_frame+1, :, :]
	hand_locations = np.zeros([tot_frames, 2], dtype = int)
	change_counter = 0

	for frame in range(0, tot_frames, 1):
		hand_locations[frame] = (frame+start_frame, hand_location_in_frame(frame, temp_data, mlist, h))
		if(hand_locations[frame][1] != hand_locations[frame-1][1]):
			change_counter = change_counter+1

	return hand_locations, change_counter

def plot_hand_location(start_frame, end_frame, data, mlist):
	r_loc, c1 = hand_location(start_frame, end_frame, data, mlist, h='R') 
	l_loc, c2 = hand_location(start_frame, end_frame, data, mlist, h='L') 
	rp = rest_pose(start_frame, end_frame, data, mlist) 
	
	x = np.arange(start_frame,end_frame)

	fig = plt.figure("HandsLocation-{}-{}".format(start_frame, end_frame), figsize=(10.5,7))
	fig.suptitle("Hands location for sign between {} and {} frame".format(start_frame, end_frame))	
	
	plt.subplot(2, 1, 1)
	plt.plot(x,r_loc[:,[1]], 'r', label='Right hand') 
	plt.plot(x,l_loc[:,[1]], 'g', label='Left hand') 
	if(len(rp)>0):
		plt.plot(x[rp],r_loc[:,[1]][rp], 'bo', label='Rest pose') 
	plt.grid(True)
	plt.xlabel("Frames")	
	plt.ylabel("Regions")

	plt.subplot(2, 2, 3)
	plt.title("R-hand location changes: {}".format(c1))
	plt.hist(r_loc[:,[1]], bins=range(15), facecolor='r', align="left")
	plt.xticks(np.arange(1, 16, step=1)) 
	plt.ylabel("Number of Frames")	
	plt.xlabel("Regions")

	plt.subplot(2, 2, 4)
	plt.title("L-hand location changes: {}".format(c2))
	plt.hist(l_loc[:,[1]], bins=range(15),  facecolor='g', align="left") 
	plt.xticks(np.arange(1, 16, step=1)) 
	plt.ylabel("Number of Frames")	
	plt.xlabel("Regions")
	
	legend = fig.legend(loc='upper right')
	# plt.show()

# compute hand velocity for the 3 axes
# returns the velocity for the 3 axes and the normilized velocity
def hand_velocity(start_frame, end_frame, data, mlist, h='R'):
	tot_frames = end_frame - start_frame
	temp_data = data[start_frame:end_frame+1, :, :]
	velocity = np.zeros([tot_frames, 3], dtype = float)
	# print(temp_data)
	hand = hand_marker(temp_data, mlist, h)

	for frame in range(0, tot_frames,1):
		velocity[frame] = hand[frame+1,:] - hand[frame,:]


	vel_norm = np.linalg.norm(velocity, axis=1)

	return velocity, vel_norm

# computes hand acceleration over normilized velocity
def hand_acceleration(velocity):
	n = len(velocity)
	acc = np.zeros([n, 1], dtype=float)

	for frame in range(0, n-1,1):
		acc[frame] = velocity[frame+1]-velocity[frame]

	return acc

# coefficients for Buterworth filter
def butter_coef(cutoff, fs, order=5):
	nyq = 0.5*fs
	Wn = cutoff/nyq # Cutoff frequency
	b,a = signal.butter(order, Wn, output='ba') #order 
	
	return b,a

# apply the filter over data
def butter_filter(data, cutoff, fs, order=5):
	# coefficients for the filter
	b,a = butter_coef(cutoff, fs, order)
	result = signal.filtfilt(b, a, data, axis = 0)
	return result

# finds the extremums in the data 
# better to use zero_crossing method of 2nd derivative 
def find_extremums(data):
	maxs = argrelextrema(data, np.greater, order=10)
	mins = argrelextrema(data, np.less, order=10)
	
	extr = np.sort(np.append(maxs, mins ))

	return extr

# creates an array (labels) 3d of frames with flags for start and end
# labels[frame][0/1][-1/0/1]
# 2nd axis is 0 if the velocity is under the threshold
def segm(vel, acc, start_frame, end_frame, data, mlist, threshold):
	# finds velocity extrema's
	vel_ex = zero_crossing(acc)
	n = len(vel_ex)
	labels = np.full([n,3], -100)
	for i in range(n):
		if(vel[vel_ex[i]] <= threshold):
			labels[i][0] = vel_ex[i]
			labels[i][1] = 0
		else: 
			labels[i][0] = vel_ex[i]
			labels[i][1] = 1
	# print(labels)
	i=1
	while(i <n-2):
		if(labels[i][1]-labels[i+1][1] < 0):
			# mark start markers
			# if there is 01 preceed by zero and hand position is near rest poseit is marked with 1
			if(labels[i-1][1] == 0 ):
				if(is_rest_pose(labels[i][0]+start_frame, data, mlist) == 1 ):
					labels[i][2] = 1
				else:
					labels[i][2] = -1
			else:
				labels[i][2] = -1
		elif(labels[i][1]-labels[i+1][1] > 0): 
			# mark end markers
			# if there is sequence like 10 followed by 0 and the hand position is near rest pose it is marked with 0
			if(labels[i+2][1] == 0):
				if(is_rest_pose(labels[i+1][0]+start_frame, data, mlist) == 1 ):
					labels[i][2] = -1
					labels[i+1][2] = 0
					i = i+1
			else:
				labels[i][2] = -1
		else:
			labels[i][2] = -1
		i = i+1
	return labels 

# loops through array of labels
# returns array 
# 	- with frames for start (where the rest pose is left) and end (where the rest pose is entered)
def get_signs_borders(labels, start_frame, end_frame):
	start=0
	end=0
	#labels[i][0/1/2]
	#0- n frames //1 - good/bad extremum // 2- start 1/end 0 /not important -1
	i = 0 
	count = 0 
	signs = []
	n = len(labels)

	while(i < n - 1):
		if(labels[i][2] == 1 ):
			start = i
			i = i + 1
			while(i < n - 1):
				if(labels[i][2] == 0 ):
					break
				if(labels[i][2] == 1 ):
					i = i - 2 
					break
				i = i +1
			end = i
			count = count + 1
			signs.append((labels[start][0], labels[end][0]))
			print("sign #{}: {}  -  {} ". format(count,labels[start][0]+start_frame, labels[end][0]+start_frame))
			
		i = i +1

	signs = np.array(signs)
	return signs

# compute everything needed for segmenting the data
# returns 2d array 
def segment_signs(start_frame, end_frame, data, mlist, fps, threshold):
	h = 'R'
	if(dominant_hand(start_frame, end_frame, data, mlist) == 1):
		h = 'R'
	elif(dominant_hand(start_frame, end_frame, data, mlist) == 2):
		h = 'L'

	velocity, vel_norm = hand_velocity(start_frame, end_frame, data, mlist, h)
	acc_norm = hand_acceleration(vel_norm)
	
	acc = butter_filter(acc_norm, 12, fps, 10)

	labels = segm(vel_norm, acc, start_frame, end_frame, data, mlist, threshold)
	signs = get_signs_borders(labels, start_frame, end_frame)

	# print(signs.shape)
	return signs

def get_real_signs(vel, labels, start_frame, end_frame):
	n = len(labels)
	start = start_frame
	end = end_frame
	###
	## maybe i should check also for vel[start_frame and end_frame]
	###
	for i in range(1,n-1):
		if(vel[labels[i-1][0]] < vel[labels[i][0]] and vel[labels[i][0]] > vel[labels[i+1][0]]):
			labels[i][2] = 0 #max
		if(vel[labels[i-1][0]] > vel[labels[i][0]] and vel[labels[i][0]] < vel[labels[i+1][0]]):
			labels[i][2] = 1 #min
	# print(labels)
	
	for i in range(1,n-1):
		if(labels[i][2] == 1):
			start = labels[i][0] 
			break;
	for i in range(n-1, 1, -1):
		if(labels[i][2] == 1):
			end = labels[i][0] 
			break;
	
	return start, end
	# return labels
	
# returns a new array with data containing only signs// without rest pose
def get_segmented_data(start_frame, end_frame, data, mlist, fps,order=0 ):
	signs_boundaries, s1 = segment_signs(start_frame, end_frame, data, mlist, fps)
	if(order==1):
		signs_boundaries = s1
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



def plot_velocity(start_frame, end_frame, vel, acc, signs, threshold):
	x = np.arange(start_frame, end_frame)

	fig1 = plt.figure("{}-{}signs-vel".format(title, count), figsize=(10.5,7))
	fig1.suptitle("Right hand velocity for sign between {} and {} frame".format(start_frame, end_frame))	

	plt.plot(x, vel, 'c', label='Normilized velocity') 
	plt.plot(x[zero_crossing], r_vel[zero_crossing], 'o')
	plt.plot(x[signs[:, 0]], r_vel[signs[:, 0]], 'rs', label = "Start")	
	plt.plot(x[signs[:, 1]], r_vel[signs[:, 1]], 'r*', label = "End")	

	# plt.plot(x[s1[:, 0]], r_vel[s1[:, 0]], 'gs', label = "Start")	
	# plt.plot(x[s1[:, 1]], r_vel[s1[:, 1]], 'g*', label = "End")

	plt.axhline(y=threshold, color='r', linestyle='-', label="Treshold")
	plt.ylabel("Velocity (mm/frame)") 
	plt.xlabel("Frames")
	plt.grid(True)
	legend = fig1.legend(loc='upper right')


	# fig2 = plt.figure("{}-{}signs-acc".format(title, count), figsize=(10.5,7))
	# fig2.suptitle("Right hand acceleration for sign between {} and {} frame".format(start_frame, end_frame))	

	# plt.plot(x, r_acc_filt, 'm', label='Filtered acceleration') 
	# plt.plot(x[zero_crossing], r_acc_filt[zero_crossing], 'o')
	# plt.plot(x[signs[:, 0]], r_acc_filt[signs[:, 0]], 'rs', label = "Start")	
	# plt.plot(x[signs[:,1]], r_acc_filt[signs[:,1]], 'r^', label = "End")	
	# plt.ylabel("Acceleration (mm/frame^2)") 
	# plt.xlabel("Frames")
	# plt.grid(True)
	# legend = fig2.legend(loc='upper right')
	
	# plt.hist(r_vel, bins='auto') 
	plt.show()

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
