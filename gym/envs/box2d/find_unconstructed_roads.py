import os
import shutil
from pdb import set_trace

from gym.envs.box2d.car_racing import CarRacing

import numpy as np
import pandas as pd

def find_roads():

    path = './touching_tracks_tests'
    # Check if dir exists TODO
    if os.path.isdir(path):
        # Remove files TODO
        shutil.rmtree(path)
    # Create dir TODO
    os.mkdir(path)

    env = CarRacing(
                allow_reverse=False, 
                show_info_panel=False,
                num_tracks=2,
                num_lanes=2,
                num_lanes_changes=0,
                num_obstacles=100,
                random_obstacle_x_position=False,
                random_obstacle_shape=False,)
    env.change_zoom()
    for j in range(100):
        env.reset()
        for i in range(len(env.tracks[0])):
            prev_tile = env.tracks[0][i-2]
            curr_tile = env.tracks[0][i-1]
            next_tile = env.tracks[0][i]
            if any(curr_tile[0] != prev_tile[1]):
                set_trace()
            elif any(curr_tile[1] != next_tile[0]):
                set_trace()
        env.screenshot(path,name=str(j),quality='high')
        np.save(path + "/info_" + str(j) + ".csv", env.info)
        np.save(path + "/track0_" + str(j) + ".csv", env.tracks[0])
        np.save(path + "/track1_" + str(j) + ".csv", env.tracks[1])

if __name__ == '__main__':
    find_roads()
