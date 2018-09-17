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

    # filename = os.path.join(dictionary_path, 'projevy_pocasi_01_ob_rh_lh_b_g_face_gaps.c3d')
    # title = 'Pocasi_01'
    # start_frame = 600
    # end_frame = 7445

    # start_frame = 1000
    # end_frame = 1710

    # filename = os.path.join(dictionary_path, 'projevy_pocasi_02_ob_rh_lh_b_g_face_gaps.c3d')
    # title = 'Pocasi_02'
    # start_frame = 608
    # end_frame = 6667

    # #end_frame = 1200
    # start_frame = 2500
    # end_frame = 3500

    filename = os.path.join(dictionary_path, 'projevy_pocasi_03_ob_rh_lh_b_g_face_gaps.c3d')
    title = 'Pocasi_03'
    start_frame = 743
    end_frame = 6680

    # filename = os.path.join(dictionary_path, 'projevy_pocasi_04_ob_rh_lh_b_g_face_gaps.c3d')
    # title = 'Pocasi_04'
    # start_frame = 600
    # end_frame = 5115

    data, marker_list, fps, total_frames = t.read_frames(filename)

    print("\n* * * {} * * *".format(title))
    print("Total lenght of the recording session: {}".format(total_frames))
    print("Frame rate is: {}".format(fps))

    ###
    #
    # change origin point to be between the hip's markers // True is to find the relative coordinates
    #
    ###
    new_origin = ['RFWT', 'LFWT', 'RBWT', 'LBWT']
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

    ###
    # compute velocity based on original trajectory
    # returns - 3 chanel (x,y,z) velocity and normilized velocity
    ###
    r_velocity, r_vel = t.hand_velocity(start_frame, end_frame, new_data, marker_list, hand)

    # compute acceleration based on normilized velocity
    r_acc = t.hand_acceleration(r_vel)
    # low pass filter of the acceleration for removing noise coused by recording techology
    r_acc_filt = t.butter_filter(r_acc, 12, fps, 10)

    # compute the median value vor velocity used as threshold for later analysis and segmentation
    median = np.median(r_vel)
    # print("Initial treshold is {}".format(median))

    ###
    # raw segmentation of whole take
    # finding start and end frame of signs (exiting rest pose - entering rest pose)
    ###
    zero_crossing = t.zero_crossing(r_acc_filt)

    labels = t.segm(r_vel, r_acc_filt, start_frame, end_frame, new_data, marker_list, median)
    signs = t.get_signs_borders(labels, start_frame, end_frame)

    count = len(signs)
    print("The number of found signs after raw segmentation is {}\n".format(count))
    # analyze velocity during rest pose for better defining the threshold used for segmentation
    rest_pose_vel = np.zeros([end_frame - start_frame])
    for i in range(0, count - 1):
        st = signs[i][1]
        en = signs[i + 1][0]
        n = en - st

        vel, vel_norm = t.hand_velocity(st + start_frame, en + start_frame, new_data, marker_list, hand)
        rest_pose_vel[st:en] = vel_norm

    tr = np.amax(rest_pose_vel)
    # print(" Refined threshold is {}", tr)

    # refined starts and ends of the signs
    labels2 = t.segm(r_vel, r_acc_filt, start_frame, end_frame, new_data, marker_list, tr)
    signs2 = t.get_signs_borders(labels2, start_frame, end_frame)

    count2 = len(signs2)

    print("The number of found signs after refined raw segmentation is {}\n".format(count2))

    ###
    #
    # analysis of each sign beased on refined raw segmentation
    #
    ###
    for sign in enumerate(signs2):
        start = sign[1][0] + start_frame
        end = sign[1][1] + start_frame

        print("\n\n*****")
        print("{}. The sign between {} - {} frame".format(sign[0] + 1, start, end))

        vel_norm = r_vel[start - start_frame:end - start_frame]
        acc = r_acc[start - start_frame:end - start_frame]
        acc_f = r_acc_filt[start - start_frame:end - start_frame]
        avg_vel = np.average(vel_norm)
        avg_acc = np.average(acc_f)

        zeros = t.zero_crossing(acc_f)

        l = t.segm(vel_norm, acc_f, start - start_frame, end - start_frame, new_data, marker_list, tr)
        real_start, real_end = t.get_real_signs(vel_norm, l, start, end)
        sign_lenght = real_end - real_start

        print()
        print("Real start and end are {}-{}".format(real_start + start, real_end + start))
        # print("Length of the sign (in frames): {}".format(sign_lenght))
        # print("Avegare velocity: {}".format(avg_vel))
        # print("Average acceleration: {}".format(avg_acc))

        diff_right_hand, diff_left_hand, diff_RWRE, diff_LWRE, right_dominant, one_hand = t.hand_displacment(real_start + start, real_end + start, new_data, marker_list)

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

        h = 'R'
        if(right_dominant == 1):
            print("- dominant hand is: Right \n")
        else:
            h = 'L'
            print("- dominant hand is: Left hand \n")

        if(one_hand == 3):
            r_loc, ch_c_r, reg_u_r = t.hand_location(real_start + start, real_end + start, new_data, marker_list, 'R')
            l_loc, ch_c_l, reg_u_l = t.hand_location(real_start + start, real_end + start, new_data, marker_list, 'L')
            print("- Right hand changes in location: {}".format(ch_c_r))
            print("- Right hand is in:")
            for reg in reg_u_r:
                print("  - region {} for {} frames ".format(reg[0], reg[1]))

            print("\n- Left hand changes in location: {}".format(ch_c_l))
            print("- Left hand is in:")
            for reg in reg_u_l:
                print("  - region {} for {} frames ".format(reg[0], reg[1]))
        else:
            loc, ch_c, reg_u = t.hand_location(real_start + start, real_end + start, new_data, marker_list, h)

            print("- Dominant hand is in:")
            for reg in reg_u:
                print("  - region {} for {} frames ".format(reg[0], reg[1]))

        x = np.arange(start, end)

        fig1 = plt.figure("{}_{}-{}-{}sign-vel".format(sign[0] + 1, title, start, end), figsize=(10.5, 7))
        fig1.suptitle("{}. Hand velocity for sign between {} and {} frame".format(sign[0] + 1, start, end))

        plt.plot(x, vel_norm, 'm', label='Normilized velocity')
        plt.plot(x, acc_f, 'c', label='Filtered acceleration')
        plt.plot(x[zeros], vel_norm[zeros], 'o')

        plt.plot(x[real_start], vel_norm[real_start], 'gs', label="Real start - {}".format(real_start + start))
        plt.plot(x[real_end], vel_norm[real_end], 'g*', label="Real end - {}".format(real_end + start))

        plt.axhline(y=tr, color='r', linestyle='-', label=" Refined Treshold")
        plt.ylabel("Velocity (mm/frame)")
        plt.xlabel("Frames")
        plt.grid(True)
        legend = fig1.legend(loc='upper right')
        t.plot_hand_location(real_start + start, real_end + start, new_data, marker_list)
        plt.show()
