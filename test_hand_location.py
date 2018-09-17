import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
sys.path.insert(0, "D:\Radi\MyProjects\MOCAP")

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

    start_frame = 1144
    end_frame = 1186

    r_loc, ch_c_r, reg_r = t.hand_location(start_frame, end_frame, new_data, marker_list, 'R')
    l_loc, ch_c_l, reg_l = t.hand_location(start_frame, end_frame, new_data, marker_list, 'L')

    print("- Right hand changes in location: {}".format(ch_c_r))
    print("- Right hand is in:")

    for reg in reg_r:
        print("  - region {} for {} frames ".format(reg[0], reg[1]))

    regions_l, count_l = np.unique(l_loc[:, [1]], return_counts=True)
    print("\n- Left hand changes in location: {}".format(ch_c_l))
    print("- Left hand is in:")
    for reg in reg_l:
        print("  - region {} for {} frames ".format(reg[0], reg[1]))

    t.plot_hand_location(start_frame, end_frame, new_data, marker_list)
    # plt.show()
