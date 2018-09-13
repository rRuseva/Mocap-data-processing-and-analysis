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
    # start_frame = 600
    # end_frame = 7445

    start_frame = 650
    end_frame = 960

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

    
    one_hand = t.is_one_handed(start_frame, end_frame, new_data, marker_list)

    message = "The sign between {} - {} frames is".format(start_frame, end_frame)
    if(one_hand == 3):
        message = message + ' two handed'
    elif(one_hand == 1):
        message = message + ' right handed'
    elif(one_hand == 2):
        message = message + " left handed"
    else:
        message = message + " with no hands"

    print(message)
