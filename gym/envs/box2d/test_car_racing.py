import os
import shutil

from gym.envs.box2d.car_racing import CarRacing

def manual_check_of_not_allowing_touching_tracks():

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
    for i in range(100):
        env.reset()
        env.screenshot(path,name=str(i),quality='high')

if __name__ == '__main__':
    print("Generating 100 random pic of maps ...")
    manual_check_of_not_allowing_touching_tracks()
    print("Done: please check if none of them has touching tracks")
