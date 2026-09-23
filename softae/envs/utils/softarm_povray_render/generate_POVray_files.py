# Copyright (c) 2025 Hehui Zheng, ETH Zurich.
# Licensed under the MIT License. See LICENSE file for details.

import numpy as np
import os
import argparse

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


def main(args):
    images_dir = os.path.join('./tmp', args.image_dir)

    # Initialize directory
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # Load elastic arm data
    if args.arm_and_target_data_path is None and args.data_dir is not None:
        arm_and_target_data_path = os.path.join(args.data_dir, 'arm_data.npz')
    elif args.arm_and_target_data_path is not None:
        arm_and_target_data_path = args.arm_and_target_data_path
    else:
        raise ValueError("Please provide a path to the arm and target data")
    arm_and_target_data = np.load(arm_and_target_data_path, allow_pickle=True)
    arm_position = arm_and_target_data["position_rod"]
    arm_radius = arm_and_target_data["radii_rod"]
    arm_n_elem = arm_and_target_data["n_elems_rod"]

    # Plot directors for rod tip and sphere
    arm_directors = arm_and_target_data["directors_rod"]
    target_directors = arm_and_target_data["directors_sphere"]
    vector_radius = 0.025 # Radius of lines
    color_order = ["SlateBlue", "LimeGreen", "OrangeRed"]

    # Load target data
    target_radius = arm_and_target_data["radii_sphere"]
    target_position = arm_and_target_data["position_sphere"]

    # If there are obstacles in the scene, load obstacle data
    if args.obstacle_data_path is None and args.data_dir is not None:
        obstacle_data_path = os.path.join(args.data_dir, 'obstacle_data.npz')
    elif args.obstacle_data_path is not None:
        obstacle_data_path = args.obstacle_data_path
    else:
        obstacle_data_path = None
    obstacles_exist = False
    if obstacle_data_path is not None and os.path.exists(obstacle_data_path):
        obstacles_exist = True
        obstacle_data = np.load(obstacle_data_path, allow_pickle=True)
        obstacle_history = obstacle_data["obstacle_history"]

        number_of_obstacles = len(obstacle_history)
        number_of_elem_obstacles = np.array(obstacle_history[0]["position_plotting"]).shape[1]

        # Obstacle positions are not changing throught the time
        obstacle_radius = np.zeros((number_of_obstacles))
        obstacle_position = np.zeros((3,number_of_elem_obstacles, number_of_obstacles))

        for i in range(number_of_obstacles):
            obstacle_radius[i] = obstacle_history[i]["radius"]
            obstacle_position[...,i] = obstacle_history[i]["position_plotting"]


    scale=3 #scales scene. Set at 3 to avoid resetting camera angles. 

    time = arm_position.shape[0]

    for k in range(time):
        file1 = open(os.path.join(images_dir, "moving_arm%02d.inc"%k),"w")

        # Arm
        file1.writelines("sphere_sweep\n{b_spline %d"% int(arm_n_elem))
        for i in range(0,arm_n_elem):
            file1.writelines(
                ",\n<%f,%f,%f>,%f" % (scale*arm_position[k][2][i] , scale*arm_position[k][0][i], scale*arm_position[k][1][i], scale*arm_radius[k][i]))
        file1.writelines("\ntexture{")
        file1.writelines("pigment{ color rgb<0.45,0.39,1>  transmit %f }"%(0.1))
        file1.writelines("finish{ phong 1 } }")
        if obstacles_exist:
            file1.writelines("scale<4,4,4> rotate<0,90,90> translate<-10,0,18>    }\n")
        else:
            file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")

        print("Arm tip position: ", arm_position[k][0][arm_n_elem-1], arm_position[k][1][arm_n_elem-1], arm_position[k][2][arm_n_elem-1])

        # Base of arm
        file1.writelines("sphere\n{")
        for i in range(0, 1):
            file1.writelines(
                "\n<%f,%f,%f>,%f" % (scale * arm_position[k][2][i] , scale * arm_position[k][0][i],
                                    scale * arm_position[k][1][i] , 1.5 * scale * arm_radius[k][i]))
        file1.writelines("\ntexture{")
        file1.writelines("pigment{ color rgb<0.75,0.75,0.75>  transmit %f }" % (0.1))
        file1.writelines("finish{ phong 1 } }")
        if obstacles_exist:
            file1.writelines("scale<4,4,4> rotate<0,90,90> translate<-10,0,18>    }\n")
        else:
            file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")

        # Target
        # file1.writelines("sphere\n{")
        # file1.writelines(
        #     "\n<%f,%f,%f>,%f" % (
        #     scale * target_position[k][2], scale * (target_position[k][0]), scale * target_position[k][1],
        #     scale * target_radius[k]))
        # file1.writelines("\ntexture{")
        # file1.writelines("pigment{ color rgb<1,0.5,0.4>  transmit %f }" % (0.1))
        # file1.writelines("finish{ phong 1 } }")
        # if obstacles_exist:
        #     file1.writelines("scale<4,4,4> rotate<0,90,90> translate<-10,0,18>    }\n")
        # else:
        #     file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")

        # print("Target position: ", target_position[k][0], target_position[k][1], target_position[k][2])

        # Observation position
        # arm_state_position = arm_and_target_data["position_rod_observed"]
        # obs_point_radius = 0.2 * scale * arm_radius[k][0]
        # # for i in [0, 5, 10, 15, 20]:
        # for i in [20]:
        #     file1.writelines("sphere\n{")
        #     file1.writelines(
        #         "\n<%f,%f,%f>,%f" % (
        #         scale * arm_state_position[k][2][i], scale * arm_state_position[k][0][i], scale * arm_state_position[k][1][i],
        #         obs_point_radius))
        #     file1.writelines("\ntexture{")
        #     # file1.writelines("pigment{ color rgb<1,0.5,0.4>  transmit %f }" % (0.1))
        #     file1.writelines("pigment{ color rgb<0,0,1>  transmit %f }" % (0.1))
        #     file1.writelines("finish{ phong 1 } }")
        #     if obstacles_exist:
        #         file1.writelines("scale<4,4,4> rotate<0,90,90> translate<-10,0,18>    }\n")
        #     else:
        #         file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")
        
        # Observation orientation
        # Drawing rod tip directors using vector
        # for idx_orientation in [4, 9, 14, 19]:
        #     for idx in range(3):
        #         arm_tip_vector_start = arm_position[k, :,idx_orientation]
        #         arm_tip_vector_end = arm_position[k, :, idx_orientation] + 5*arm_radius[k,idx_orientation] * arm_directors[k, idx, :, idx_orientation] # directors(time, vectorstart, vectorend, element index)
        #         file1.writelines("object\n{")
        #         file1.writelines("Vector\n(")
        #         file1.writelines(
        #             "\n<%f,%f,%f>,   <%f, %f, %f>, %f)" % (
        #                 scale * (arm_tip_vector_start[2]),
        #                 scale * (arm_tip_vector_start[0]),
        #                 scale * (arm_tip_vector_start[1]),
        #                 scale * (arm_tip_vector_end[2] ),
        #                 scale * (arm_tip_vector_end[0] ),
        #                 scale * (arm_tip_vector_end[1] ),
        #                 vector_radius,
        #                 ))
        #         file1.writelines("\ntexture{")
        #         file1.writelines(str("pigment{ color "+color_order[idx]+" transmit %f }") % (0.1))
        #         file1.writelines("finish{ phong 1 } }")
        #         file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")



        # Obstacles
        if obstacles_exist:
            for j in range(0, number_of_obstacles):
            # for j in obstacles_to_be_plotted:
                file1.writelines("sphere_sweep\n{b_spline %d" % int(number_of_elem_obstacles))
                for i in range(0, number_of_elem_obstacles):
                    file1.writelines(
                        ",\n<%f,%f,%f>,%f" % (
                        scale * obstacle_position[2][i][j], scale * obstacle_position[0][i][j], scale * obstacle_position[1][i][j],
                        scale * obstacle_radius[j]))
                file1.writelines("\ntexture{")
                file1.writelines("pigment{ color LimeGreen  transmit %f }" % (0.1))
                file1.writelines("finish{ phong 1 } }")
                file1.writelines("scale<4,4,4> rotate<0,90,90> translate<-10,0,18>    }\n")

        file1.close()
        file2 = open(os.path.join(images_dir, "moving_arm%02d.pov"%k),"w")

        current_dir = os.path.dirname(os.path.realpath(__file__))
        camera_file = os.path.join(current_dir, "camera_position_0.inc")
        # Use forward slashes in paths for compatibility with POV-Ray
        camera_file = camera_file.replace("\\", "/")
        file2.writelines(f'#include "{camera_file}"\n')

        file2.writelines("#include \"moving_arm%02d.inc\"\n"%k)
        file2.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None, help='path of the data directory')
    parser.add_argument('--image_dir', type=str, default='', help='path of the povray file directory')
    parser.add_argument('--arm_and_target_data_path', type=str, default=None, help='path of the arm and target data')
    parser.add_argument('--obstacle_data_path', type=str, default=None, help='path of the obstacle data')

    args = parser.parse_args()
    main(args)
