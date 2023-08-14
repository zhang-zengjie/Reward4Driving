import pytest
from gym.envs.box2d.car_racing import CarRacing

class TestCarRacing(object):
    def test_one_track(self):
        env = CarRacing()

        # Tracks should not exist before any reset
        with pytest.raises(AttributeError):
            env.tracks

        env.reset()
        assert len(env.tracks) == 1

        env.close()

    def test_two_tracks(self):
        env = CarRacing(num_tracks=2)

        env.reset()
        assert len(env.tracks) == 2

        env.close()

    def test_two_lanes_with_no_lane_changes(self):
        env = CarRacing(num_lanes=2,num_lanes_changes=0)
        env.reset()
        assert env._get_extremes_of_position(0,border=1) == (-3-1/3,3+1/3)

        assert env._get_extremes_of_position(1,border=0) == (-6-2/3,+6+2/3)

        env.close()
        del env

    def test_screenshot(self,tmpdir):
        # TODO use tempdir to save screenshots
        pass
