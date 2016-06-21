"""
Connect a BECCA brain to a world and run them.

To use this module as a top level script:
    1. Select the World that the Brain will be placed in.
    2. Make sure the appropriate import line is included and
        uncommented below.
    3. Run from the command line.
        > python tester.py
"""

from __future__ import print_function
import cProfile
import pstats

import numpy as np

# Worlds from the benchmark
#from becca.worlds.base_world import World
from becca.worlds.grid_1D import World
#from becca.worlds.grid_1D_delay import World
#from becca.worlds.grid_1D_chase import World
#from becca.worlds.grid_1D_ms import World
#from becca.worlds.grid_1D_noise import World
#from becca.worlds.grid_2D import World
#from becca.worlds.grid_2D_dc import World
#from becca.worlds.image_1D import World
#from becca.worlds.image_2D import World
#from becca.worlds.fruit import World

# If you want to run a world of your own, add the appropriate line here
#from worlds.hello import World
#from becca_world_chase_ball.chase import World

from becca.core.brain import Brain

def test(world_class, testing_period=2e4):
    """
    Test the brain's performance on a world.

    Parameters
    ----------
    world_class : World
        The class containing the BECCA-compatible world that the
        brain will be receiving sensor and reward information from and
        sending action commands to.
    testing_period : int, optional
        The number of time steps to test the brain
        on the current world.  The default is 10,000.

    Returns
    -------
    performance : float
        The average reward per time step during the testing period.
    world.name : str
        The name of the world that was run.
    """
    world = world_class(lifespan=testing_period)
    performance = run(world)
    print('Performance is: {0:.3}'.format(performance))
    return performance, world.name

def run(world, restore=False):
    """
    Run BECCA with a world.

    Connects the brain and the world together and runs them for as long
    as the world dictates.

    Parameters
    ----------
    restore : bool, optional
        If ``restore`` is True, try to restore the brain from a previously saved
        version, picking up where it left off.
        Otherwise it create a new one. The default is False.

    Returns
    -------
    performance : float
        The performance of the brain over its lifespan, measured by the
        average reward it gathered per time step.
    """
    brain_name = '{0}_brain'.format(world.name)
    brain = Brain(world.num_sensors, world.num_actions, brain_name=brain_name)
    if restore:
        brain = brain.restore()
    # Start at a resting state.
    actions = np.zeros(world.num_actions)
    sensors, reward = world.step(actions)
    # Repeat the loop through the duration of the existence of the world:
    # sense, act, repeat.
    while world.is_alive():
        actions = brain.sense_act_learn(sensors, reward)
        sensors, reward = world.step(actions)
        world.visualize(brain)
    performance = brain.report_performance()
    return performance

def profile():
    """
    Profile the brain's performance on the selected world.
    """
    profiling_lifespan = 1e4
    print('Profiling BECCA\'s performance...')
    command = 'run(World(lifespan={0}), restore=True)'.format(
        profiling_lifespan)
    cProfile.run(command, 'tester.profile')
    profile_stats = pstats.Stats('tester.profile')
    profile_stats.strip_dirs().sort_stats('time', 'cumulative').print_stats(30)
    print('   View at the command line with')
    print(' > python -m pstats tester.profile')

if __name__ == '__main__':
    # To profile BECCA's performance with world, set profile_flag to True.
    PROFILE_FLAG = False
    if PROFILE_FLAG:
        profile()
    else:
        run(World(lifespan=1e8), restore=True)
