import numpy as np
import tools as t


def segmentation(start_frame, end_frame, new_data, marker_list, hand, fps):
    ###
    # compute velocity based on original trajectory
    # returns - 3 chanel (x,y,z) velocity and normilized velocity
    ###
    velocity, vel_norm = t.hand_velocity(start_frame, end_frame, new_data, marker_list, hand)
    # compute the median value vor velocity used as threshold for later analysis and segmentation
    median = np.median(vel_norm)
    # print("Initial treshold is {}".format(median))

    # compute acceleration based on normilized velocity
    acc = t.hand_acceleration(vel_norm)
    # low pass filter of the acceleration for removing noise coused by recording techology
    acc_filt = t.butter_filter(acc, 12, fps, 10)

    ###
    # raw segmentation of whole take
    # finding start and end frame of signs (exiting rest pose - entering rest pose)
    ###
    zero_crossing = t.zero_crossing(acc_filt)

    labels = t.segm(vel_norm, acc_filt, start_frame, end_frame, new_data, marker_list, median)
    signs = t.get_signs_borders(labels, start_frame, end_frame)

    count = len(signs)
    print("The number of found signs after raw segmentation is {}\n".format(count))

    threshold = t.rest_pose_analyzis(signs, start_frame, end_frame, new_data, marker_list, hand)

    # refined starts and ends of the signs
    labels2 = t.segm(vel_norm, acc_filt, start_frame, end_frame, new_data, marker_list, threshold)
    signs2 = t.get_signs_borders(labels2, start_frame, end_frame)

    count2 = len(signs2)

    print("The number of found signs after refined raw segmentation is {}\n".format(count2))

    return signs2, count2, vel_norm, acc, acc_filt, threshold
