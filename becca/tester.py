"""
Connect a BECCA brain to a world and run them.

To use this module as a top level script:
    1. Select the World that the Brain will be placed in.
    2. Make sure the appropriate import line is included and
        uncommented below.
    3. Run from the command line.
        > python tester.py

Usage
-----
Test BECCA on the grid1D.py world.
> python tester.py grid1D

Test BECCA on the suite of all test worlds.
> python tester.py all

Profile BECCA on the image2D.py world.
> python tester.py image2D --profile 
"""

from __future__ import print_function
import argparse
import cProfile
import matplotlib.pyplot as plt
import pstats

import numpy as np


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
        on the current world.

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


def suite():
    """
    Run all the worlds in the benchmark and tabulate their performance.
    """
    # Import the suite of test worlds
    from becca.worlds.grid_1D import World as World_grid_1D
    from becca.worlds.grid_1D_chase import World as World_grid_1D_chase
    from becca.worlds.grid_1D_delay import World as World_grid_1D_delay
    from becca.worlds.grid_1D_ms import World as World_grid_1D_ms
    from becca.worlds.grid_1D_noise import World as World_grid_1D_noise
    from becca.worlds.grid_2D import World as World_grid_2D
    from becca.worlds.grid_2D_dc import World as World_grid_2D_dc
    from becca.worlds.image_1D import World as World_image_1D
    from becca.worlds.image_2D import World as World_image_2D
    from becca.worlds.fruit import World as World_fruit

    performance = []
    performance.append(test(World_grid_1D))
    performance.append(test(World_grid_1D_delay))
    performance.append(test(World_grid_1D_chase))
    performance.append(test(World_grid_1D_ms))
    performance.append(test(World_grid_1D_noise))
    performance.append(test(World_grid_2D))
    performance.append(test(World_grid_2D_dc))
    performance.append(test(World_image_1D))
    performance.append(test(World_image_2D))
    performance.append(test(World_fruit))
    print('Individual test world scores:')
    scores = []
    for score in performance:
        print('    {0:.2}, {1}'.format(score[0], score[1]))
        scores.append(score[0])
    print('Overall test suite score: {0:.2}'.format(np.mean(scores)))

    # Block the program, displaying all plots.
    # When the plot windows are closed, the program closes.
    plt.show()


def profile(World):
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
    # Build the command line parser.
    parser = argparse.ArgumentParser(description='Test BECCA on some toy worlds.')
    parser.add_argument('world', default='all',
                        help=' '.join(['The test world to run.',
                                       'Choose by name or number:', 
                                       '1) grid1D,', 
                                       '2) grid1D_chase,',
                                       '3) grid1D_delay,',
                                       '4) grid1D_ms,',
                                       '5) grid1D_noise,',
                                       '6) grid2D,',
                                       '7) grid2D_dc,',
                                       '8) image1D,',
                                       '9) image2D,',
                                       '10) fruit,',
                                       '0) all',
                                       'Default value is all.']))
    parser.add_argument('-p', '--profile', action='store_true')
    args = parser.parse_args()
    print(args)

    if args.world == 'grid1D' or args.world == '1': 
        from becca.worlds.grid_1D import World as World_grid_1D
        World = World_grid_1D
    elif args.world == 'grid1D_chase' or args.world == '2':
        from becca.worlds.grid_1D_chase import World as World_grid_1D_chase
        World = World_grid_1D_chase
    elif args.world == 'grid1D_delay' or args.world == '3':
        from becca.worlds.grid_1D_delay import World as World_grid_1D_delay
        World = World_grid_1D_delay
    elif args.world == 'grid1D_ms' or args.world == '4':
        from becca.worlds.grid_1D_ms import World as World_grid_1D_ms
        World = World_grid_1D_ms
    elif args.world == 'grid1D_noise' or args.world == '5':
        from becca.worlds.grid_1D_noise import World as World_grid_1D_noise
        World = World_grid_1D_noise
    elif args.world == 'grid2D' or args.world == '6':
        from becca.worlds.grid_2D import World as World_grid_2D
        World = World_grid_2D
    elif args.world == 'grid2D_dc' or args.world == '7':
        from becca.worlds.grid_2D_dc import World as World_grid_2D_dc
        World = World_grid_2D_dc
    elif args.world == 'image1D' or args.world == '8':
        from becca.worlds.image_1D import World as World_image_1D
        World = World_image_1D
    elif args.world == 'image2D' or args.world == '9':
        from becca.worlds.image_2D import World as World_image_2D
        World = World_image_2D
    elif args.world == 'fruit' or args.world == '10':
        from becca.worlds.fruit import World as World_fruit
        World = World_fruit
    else:
        args.world = 'all'

    # To profile BECCA's performance with world, set profile_flag to True.
    #PROFILE_FLAG = False
    #if PROFILE_FLAG:
    if args.world == 'all':
        suite()
    elif args.profile:
        profile(World)
    else:
        performance = run(World(lifespan=5e4), restore=True)
        print('performance:', performance)
