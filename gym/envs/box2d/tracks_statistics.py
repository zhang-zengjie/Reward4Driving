import os
import shutil

from gym.envs.box2d.car_racing import CarRacing

def run():

    env = CarRacing(
                allow_reverse=False, 
                show_info_panel=False,
                num_tracks=2,
                num_lanes=2,
                num_lanes_changes=0,
                num_obstacles=100,
                random_obstacle_x_position=False,
                random_obstacle_shape=False,)

    x_count = 0
    t_count = 0

    num_tracks = 300
    for i in range(num_tracks):
        env.reset()
        print("track %i / %i" % (i, num_tracks))
        if (env.info['x'] == True).sum() > 0: x_count += 1
        if (env.info['t'] == True).sum() > 0: t_count += 1

    print('')
    i += 1
    print('num of tracks:', str(num_tracks))
    print('time took per track is {0}s'.format(1))
    print('{0}% of tracks have x intersections'.format(x_count/num_tracks))
    print('{0}% of tracks have t intersections'.format(t_count/num_tracks))

if __name__ == '__main__':
    run()
