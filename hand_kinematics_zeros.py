import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, "D:\Radi\MyProjects\MOCAP")

import tools as t

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

    data, marker_list, fps, tot_frames = t.read_frames(filename)

    # change origin point to be between the hip's markers // True is to find the relative coordinates
    new_origin = ['RFWT', 'LFWT', 'RBWT']
    new_data = t.change_origin_point(data, new_origin, marker_list, True)

    ###
    #
    # checks for dominant hand
    #
    ##
    right_dominant = t.dominant_hand(start_frame, end_frame, data, marker_list)
    hand = 'R'
    if(right_dominant == 1):
        print("- more active hand is: Right \n")
    else:
        print("- more active hand is: Left hand \n")
        hand = 'L'

    r_velocity, r_vel = t.hand_velocity(start_frame, end_frame, new_data, marker_list, hand)
    median = np.median(r_vel)
    average = np.average(r_vel)
    mode = stats.mode(r_vel)

    r_acc = t.hand_acceleration(r_vel)
    r_acc_filt = t.butter_filter(r_acc, 12, fps, 10)
    zero_crossing = t.zero_crossing(r_acc_filt)

    x = np.arange(start_frame, end_frame)

    fig1 = plt.figure("{}-signs-vel".format(title), figsize=(10.5, 7))
    fig1.suptitle("Right hand velocity for sign between {} and {} frame".format(start_frame, end_frame))

    plt.subplot(2, 1, 1)

    plt.plot(x, r_vel, 'c', label='Normilized velocity')
    plt.plot(x[zero_crossing], r_vel[zero_crossing], 'o')

    plt.axhline(y=median, color='r', linestyle='-', label="Treshold - median")

    plt.axhline(y=average, color='g', linestyle='-', label="Treshold - average")
    plt.axhline(y=mode[0], color='b', linestyle='-', label="Treshold - mode")

    plt.ylabel("Velocity (mm/frame)")
    plt.xlabel("Frames")
    plt.grid(True)
    legend = fig1.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.plot(x, r_acc_filt, 'm', label='Filtered acceleration')
    plt.plot(x[zero_crossing], r_acc_filt[zero_crossing], 'o')
    plt.ylabel("Acceleration (mm/frame^2)")
    plt.xlabel("Frames")
    plt.grid(True)

    plt.show()
