import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, "D:\Radi\MyProjects\MOCAP")
# import mocap_tools

import tools as t

if __name__ == "__main__":
    dictionary_path = ''

    # filename = os.path.join(dictionary_path, 'projevy_pocasi_01_ob_rh_lh_b_g_face_gaps.c3d')
    # title = 'Pocasi_01'
    # start_frame = 600
    # end_frame = 7445

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

    start_frame = 1
    end_frame = 5368

    data, marker_list, fps, tot_frames = t.read_frames(filename)

    # change origin point to be between the hip's markers // True is to find the relative coordinates
    new_origin = ['RFWT', 'LFWT', 'RBWT']
    new_data = t.change_origin_point(data, new_origin, marker_list, True)

    r_velocity, r_vel = t.hand_velocity(start_frame, end_frame, new_data, marker_list, 'R')

    r_acc = t.hand_acceleration(r_vel)
    cutoff = 12

    r_acc_filt = t.butter_filter(r_acc, cutoff, fps, 10)
    r_acc_filt2 = t.butter_filter(r_acc, 2, fps, 10)
    r_acc_filt4 = t.butter_filter(r_acc, 4, fps, 10)
    r_acc_filt10 = t.butter_filter(r_acc, 10, fps, 10)
    zero_crossing = t.zero_crossing(r_acc_filt)

    x = np.arange(start_frame, end_frame)

    fig2 = plt.figure("{}-hand_acc".format(title), figsize=(10.5, 7))

    fig2.suptitle("Hand acceleration")
    plt.plot(x, r_acc, 'c', label='Acceleration ')

    # plt.plot(x, r_acc_filt2, 'y', label='Filtered acceleration / coff = 2 ')

    # plt.plot(x, r_acc_filt4, 'g', label='Filtered acceleration / coff = 4 ')

    # plt.plot(x, r_acc_filt10, 'b', label='Filtered acceleration / coff = 10 ')

    plt.plot(x, r_acc_filt, 'm', label='Filtered acceleration / coff = {}'.format(cutoff))

    plt.plot(x[zero_crossing], r_acc_filt[zero_crossing], 'o')
    plt.ylabel("Acceleration (mm/frame^2)")
    plt.xlabel("Frames")
    plt.grid(True)
    legend = fig2.legend(loc='upper right')

    plt.show()
