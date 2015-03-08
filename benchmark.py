"""
benchmark 0.6.1

A suite of worlds to characterize the performance of BECCA variants.
Other agents may use this benchmark as well, as long as they have the 
same interface. (See BECCA documentation for a detailed specification.)
In order to facilitate apples-to-apples comparisons between agents, the 
benchmark will be version numbered.

Run at the command line as a script with no argmuments:
> python benchmark.py

For N_RUNS = 11, Becca 0.6.1 scored 75
"""
import matplotlib.pyplot as plt
import numpy as np
import tester
from core.agent import Agent
from worlds.grid_1D import World as World_grid_1D
from worlds.grid_1D_delay import World as World_grid_1D_delay
from worlds.grid_1D_ms import World as World_grid_1D_ms
from worlds.grid_1D_noise import World as World_grid_1D_noise
from worlds.grid_2D import World as World_grid_2D
from worlds.grid_2D_dc import World as World_grid_2D_dc
from worlds.image_1D import World as World_image_1D
from worlds.image_2D import World as World_image_2D

def main():
    N_RUNS = 11
    benchmark_lifespan = 1e4
    overall_performance = []
    # Run all the worlds in the benchmark and tabulate their performance
    for i in range(N_RUNS):
        performance = []
        world = World_grid_1D(lifespan=benchmark_lifespan)
        performance.append(tester.test(world, show=False))
        world = World_grid_1D_delay(lifespan=benchmark_lifespan)
        performance.append(tester.test(world, show=False))
        world = World_grid_1D_ms(lifespan=benchmark_lifespan)
        performance.append(tester.test(world, show=False))
        world = World_grid_1D_noise(lifespan=benchmark_lifespan)
        performance.append(tester.test(world, show=False))
        world = World_grid_2D(lifespan=benchmark_lifespan)
        performance.append(tester.test(world, show=False))
        world = World_grid_2D_dc(lifespan=benchmark_lifespan)
        performance.append(tester.test(world, show=False))
        world = World_image_1D(lifespan=benchmark_lifespan)
        performance.append(tester.test(world, show=False))
        world = World_image_2D(lifespan=benchmark_lifespan)
        performance.append(tester.test(world, show=False))

        print "Individual benchmark scores: " , performance
        total = 0
        for val in performance:
            total += val
        mean_performance = total / len(performance)
        overall_performance.append(mean_performance)
        print "Overall benchmark score, ", i , "th run: ", mean_performance 
    print "All overall benchmark scores: ", overall_performance 
    
    #  Find the median benchmark score
    print "Typical performance score: ", np.median(np.array(overall_performance))
    
    # Block the program, displaying all plots.
    # When the plot windows are closed, the program closes.
    plt.show()
    
if __name__ == '__main__':
    main()
