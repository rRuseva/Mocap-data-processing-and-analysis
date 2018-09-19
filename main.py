import numpy as np
import matplotlib.pyplot as plt
import argparse
import tools as t
import read_input as ri
import segmentation as s


if __name__ == "__main__":
    # load whole take file with start/end frame without T-poses

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='fname', type=int)
    args = parser.parse_args()
    fname = args.fname
    # fname = 3

    title, start_frame, end_frame, new_data, marker_list, fps, total_frames, hand = ri.read_input(fname)

    signs, count, velocity_norm, acceleration, acc_filt, threshold = s.segmentation(start_frame, end_frame, new_data, marker_list, hand, fps)

    ###
    # analysis of each sign beased on refined raw segmentation
    ###
    for sign in enumerate(signs):
        start = sign[1][0] + start_frame
        end = sign[1][1] + start_frame

        print("\n\n*****")
        print("{}. The sign between {} - {} frame".format(sign[0] + 1, start, end))

        vel_norm = velocity_norm[start - start_frame:end - start_frame]
        acc = acceleration[start - start_frame:end - start_frame]
        acc_f = acc_filt[start - start_frame:end - start_frame]
        avg_vel = np.average(vel_norm)
        avg_acc = np.average(acc_f)

        zeros = t.zero_crossing(acc_f)

        labels = t.segm(vel_norm, acc_f, start - start_frame, end - start_frame, new_data, marker_list, threshold)
        real_start, real_end = t.get_real_signs(vel_norm, labels, start, end)
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

        plt.axhline(y=threshold, color='r', linestyle='-', label=" Refined Treshold")
        plt.ylabel("Velocity (mm/frame)")
        plt.xlabel("Frames")
        plt.grid(True)
        legend = fig1.legend(loc='upper right')
        t.plot_hand_location(real_start + start, real_end + start, new_data, marker_list)
    plt.show()
