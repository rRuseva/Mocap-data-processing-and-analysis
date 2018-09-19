import os
import sys
import tools as t
sys.path.insert(0, "D:\Radi\MyProjects\MOCAP")


def read_input(fname):
    dictionary_path = ''
###
# chose which file to load whole take file with start/end frame without T-poses
###
    if(fname == 1):
        filename = os.path.join(dictionary_path, 'projevy_pocasi_01_ob_rh_lh_b_g_face_gaps.c3d')
        title = 'Pocasi_01'
        start_frame = 600
        end_frame = 7445
    elif(fname == 2):
        filename = os.path.join(dictionary_path, 'projevy_pocasi_02_ob_rh_lh_b_g_face_gaps.c3d')
        title = 'Pocasi_02'
        start_frame = 608
        end_frame = 6667
    elif(fname == 3):
        filename = os.path.join(dictionary_path, 'projevy_pocasi_03_ob_rh_lh_b_g_face_gaps.c3d')
        title = 'Pocasi_03'
        start_frame = 743
        end_frame = 6680
    else:
        filename = os.path.join(dictionary_path, 'projevy_pocasi_04_ob_rh_lh_b_g_face_gaps.c3d')
        title = 'Pocasi_04'
        start_frame = 600
        end_frame = 5115

    data, marker_list, fps, total_frames = t.read_frames(filename)

    print("\n* * * {} * * *".format(title))
    print("Total lenght of the recording session: {}".format(total_frames))
    print("Frame rate is: {}".format(fps))

###
# change origin point to be between the hip's markers // True is to find the relative coordinates
###
    new_origin = ['RFWT', 'LFWT', 'RBWT', 'LBWT']
    new_data = t.change_origin_point(data, new_origin, marker_list, True)

    ###
# checks for dominant hand
##
    right_dominant = t.dominant_hand(start_frame, end_frame, data, marker_list)
    hand = 'R'
    if(right_dominant == 1):
        print("- more active hand is: Right \n")
    else:
        print("- more active hand is: Left hand \n")
        hand = 'L'

    return title, start_frame, end_frame, new_data, marker_list, fps, total_frames, hand
