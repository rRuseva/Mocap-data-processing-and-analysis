import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, "D:\Radi\Radi RU\4ti kurs\2sm-WBU\MOCAP\Python\mocap")

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

    data, marker_list, fps, total_frames = t.read_frames(filename)

    print("\n* * * {} * * *".format(title))

    ###
    #
    # change origin point to be between the hip's markers // True is to find the relative coordinates
    #
    ###
    new_origin = ['RFWT', 'LFWT', 'RBWT', 'LBWT']
    new_data = t.change_origin_point(data, new_origin, marker_list, True)

    # # weather
    one_hand = t.is_one_handed(650, 960, data, marker_list)   # 2 handed

    # one_hand = t.is_one_handed(1074, 1345, data, marker_list) #2 handed

    # one_hand = t.is_one_handed(1345, 1672, data, marker_list) #2 handed

    # one_hand = t.is_one_handed(1690, 2017, data, marker_list) #2 handed

    # one_hand = t.is_one_handed(2017, 2358, data, marker_list) # 2 handed

    # one_hand = t.is_one_handed(2485, 2819, data, marker_list) # 2 handed

    # one_hand = t.is_one_handed(2819, 3178, data, marker_list) # 2 handed

    # one_hand = t.is_one_handed(3178, 3776, data, marker_list) #??? is it one or two handed // GIVES IT AS RIGHT HANDED

    # one_hand = t.is_one_handed(3776, 4064, data, marker_list) #??? is it one or two handed // GIVES IT AS RIGHT HANDED

    # one_hand = t.is_one_handed(4064, 4414, data, marker_list) # 2 handed

    # one_hand = t.is_one_handed(4414, 4798, data, marker_list) # 2 handed

    # one_hand = t.is_one_handed(4798, 5053, data, marker_list) # 1 handed RIGHT

    # one_hand = t.is_one_handed(5053, 5428, data, marker_list) # righ handed

    # one_hand = t.is_one_handed(5428, 5789, data, marker_list) # righ handed

    # one_hand = t.is_one_handed(5789, 6114, data, marker_list) # right handed

    # one_hand = t.is_one_handed(6114, 6479, data, marker_list) # 2 handed

    # one_hand = t.is_one_handed(6479, 6800, data, marker_list) # 2 handed

    # one_hand = t.is_one_handed(6800, 7122, data, marker_list) # 2 handed

    # one_hand = t.is_one_handed(7122, 7439, data, marker_list) # 2 handed

    # #test combine two signs
    # one_hand = t.is_one_handed(4798, 5428, data, marker_list) # 1 hand // GIVES IT AS RIGHT HANDED

    # one_hand = t.is_one_handed(5428, 6114, data, marker_list) # 1 hand // GIVES IT AS RIGHT HANDED

    # #tes whole take
    # one_hand = t.is_one_handed(205, 7439, data, marker_list) # //GIVES IT AS 2 HANDED, RIGH DOMIMANT HAND

    message = "The sign is \n-"
    if(one_hand == 3):
        message = message + ' two handed'
    elif(one_hand == 1):
        message = message + ' right handed'
    elif(one_hand == 2):
        message = message + " left handed"
    else:
        message = message + " with no hands"

    print(message)
