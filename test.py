import os
import sys
import numpy as np
sys.path.insert(0, "D:\Radi\Radi RU\4ti kurs\2sm-WBU\MOCAP\Python\mocap")
import mocap_tools

import tools as t
import tools_new as tn

if __name__ == "__main__":
	dictionary_path = ''
	
	filename = os.path.join(dictionary_path, 'projevy_pocasi_01_ob_rh_lh_b_g_face_gaps.c3d')
	title = 'Pocasi_01'
	start_frame = 600
	end_frame = 7445


	#print(mlist)
    # print(marker_index(mlist, 'RIDX3'))
    #marker_index(mlist, 'ROW')
	
	data, marker_list, fps = t.read_frames(filename)
	# print(data[1:2, :, 0:3])

	# new_origin = ['RFWT', 'LFWT', 'RBWT', 'LBWT']
	# new_data = t.change_origin_point(data, new_origin, marker_list, True)
	# # # print(new_data[1:2, :, 0:3])

	# abs_data = t.change_origin_point(data, [], marker_list, False)
	# # print(abs_data[1:2, :, 0:3])

	# diff = np.zeros((int(data.shape[0]), int(data.shape[1]), 3) )
	# for frame, (point) in enumerate(data):
	# 	for marker in range(data.shape[1]):
	# 		diff[frame, marker, :]=new_data[frame, marker, :] - data[frame, marker, :]
	# 		#print(diff)

	# 		if((abs_data[frame, marker, :]!=data[frame, marker, :]).any()):
	# 			print('false')
	# 			break;

	# print(diff[1:2, :, :])

			
	# print(marker_list)
	# m=t.marker_index(marker_list,'ORIGIN')
	# print(m)
	# print(data[1:2, 0, 0:3])
	# print(data[1:2, m, 0:3])

	# print(fps)oscillates
	#print(marker_name(marker_list, 0))

	# print(data[205:650, 0:1, :])
	# starts from T-pose // keyboard typing
	# t.is_one_handed(205,650, data, marker_list) # two handed
	# t.hand_speed(205,640, new_data, marker_list, 'R')
	# t.hand_speed(205,640, new_data, marker_list, 'L')
	# t.plot_hands_location(205,640, new_data, marker_list)

	# weather
	# t.is_one_handed(650,960, data, marker_list)	# 2 handed
	# t.plot_hand_trajectory(650,960, data, marker_list)

	# t.hand_speed(650,960, data, marker_list, 'R')
	# t.hand_speed(650,960, data, marker_list, 'L')
	# t.plot_hands_location(650,960, data, marker_list)

	# t.is_one_handed(1074, 1345, data, marker_list) #2 handed 

	# t.is_one_handed(1345, 1672, data, marker_list) #2 handed

	# t.is_one_handed(1690, 2017, data, marker_list) #2 handed 

	# t.is_one_handed(2017, 2358, data, marker_list) # 2 handed

	# t.is_one_handed(2485, 2819, data, marker_list) # 2 handed

	# t.is_one_handed(2819, 3178, data, marker_list) # 2 handed

	# t.is_one_handed(3178, 3776, data, marker_list) #??? is it one or two handed // GIVES IT AS RIGHT HANDED
	
	# t.is_one_handed(3776, 4064, data, marker_list) #??? is it one or two handed // GIVES IT AS RIGHT HANDED

	# t.is_one_handed(4064, 4414, data, marker_list) # 2 handed

	# t.is_one_handed(4414, 4798, data, marker_list) # 2 handed

	# t.is_one_handed(4798, 5053, data, marker_list) # 1 handed RIGHT

	# t.is_one_handed(5053, 5428, data, marker_list) # righ handed

	# t.is_one_handed(5428, 5789, data, marker_list) # righ handed

	# t.is_one_handed(5789, 6114, data, marker_list) # right handed

	# t.is_one_handed(6114, 6479, data, marker_list) # 2 handed

	# t.is_one_handed(6479, 6800, data, marker_list) # 2 handed

	# t.is_one_handed(6800, 7122, data, marker_list) # 2 handed 

	# t.is_one_handed(7122, 7439, data, marker_list) # 2 handed

		
	# #test combine two signs
	# t.is_one_handed(4798, 5428, data, marker_list) # 1 hand // GIVES IT AS RIGHT HANDED

	# t.is_one_handed(5428, 6114, data, marker_list) # 1 hand // GIVES IT AS RIGHT HANDED

	# #tes whole take 
	# t.is_one_handed(205, 7439, data, marker_list) # //GIVES IT AS 2 HANDED, RIGH DOMIMANT HAND
	


	# #tests for hand_location_in_frame()
	# print('1')
	# print(hand_location_in_frame(1, data, marker_list, 'R'))
	# print(hand_location_in_frame(1, data, marker_list, 'L'))
	
	# print('400 keybord')
	# print(hand_location_in_frame(400, data, marker_list, 'R'))
	# print(hand_location_in_frame(400, data, marker_list, 'L'))

	# print('987 rest pose')
	# print(hand_location_in_frame(987, data, marker_list, 'R'))
	# print(hand_location_in_frame(987, data, marker_list, 'L'))

	# print('841 ')
	# print(hand_location_in_frame(841, data, marker_list, 'R'))
	# print(hand_location_in_frame(841, data, marker_list, 'L'))
	
	# print('1149 in front of chest')
	# print(hand_location_in_frame(1149, data, marker_list, 'R'))
	# print(hand_location_in_frame(1149, data, marker_list, 'L'))
	
	# print('1491 cut in front of chest')
	# print(hand_location_in_frame(1491, data, marker_list, 'R'))
	# print(hand_location_in_frame(1491, data, marker_list, 'L'))
	
	# print('3642 temperature sign ')
	# print(hand_location_in_frame(3642, data, marker_list, 'R'))
	# hand_location(3500,3804, data, marker_list, 'R')
	# plot_hands_location(3500,3804, data, marker_list)

	# print('2843 - 3213 // symetrical sign')
	# plot_hands_location(2843,3213, data, marker_list)

	# print('205 - 640 // keyboard typing')
	# plot_hands_location(205, 640, data, marker_list)


	# print('1086 - 1373 // ')
	# plot_hands_location(1086, 1373, data, marker_list)

	# print('1669 - 2000 // ')
	# plot_hands_location(1669, 2000, data, marker_list)

	# print('682 - 1012 // circles')
	# plot_hands_location(682, 1012, data, marker_list)

	# n_d = change_zero_point(data, zero)
	# print(data[1:690, 0:1, :])
	# print(n_d[1:690, 0:1, :])

	# print('205 - 7574 // whole take')
	# plot_hands_location(205,7574, data, marker_list)




	
	
	# t.hand_speed(205,640, data, marker_list, 'R')
	# t.plot_hands_location(205,640, data, marker_list)
	



	# print('205// T-pose')
	# t.wrist_orientation_frame(205, data, marker_list, 'R')
	# t.wrist_orientation_frame(205, data, marker_list, 'L')

	# t.palm_orientation_frame(205, data, marker_list, 'R')

	t.plot_hand_movement(start_frame, end_frame, data, marker_list)
	tn.plot_hand_movement(start_frame, end_frame, data, marker_list)
