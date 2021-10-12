import numpy as np
import pandas as pd

from itertools import permutations, combinations
from importlib import reload
from multiprocessing import Pool
from time import time

import balancers
import tools


# Trying a loop with multiprocessing
outcomes = ['yes', 'no', 'maybe']
groups2 = ['a', 'b']
groups3 = ['a', 'b', 'c']
losses = ['micro', 'macro']
goals = ['odds', 'opportunity', 'strict']

# Setting up the combinations of situations
bias_types = ['low', 'medium', 'high']
out_types = ['balanced', 'one_rare', 'two_rare']
group_types = ['no_minority', 'slight_minority', 'strong_minority']

sits = [[[[g, o, p] for p in bias_types]
             for o in out_types]
            for g in group_types]
sits = tools.flatten([l for l in tools.flatten(sits)])

# Trying a run for a 3-class 2-group problem
p23 = {'groups': {
        'no_minority': np.array([.5, .5]),
        'slight_minority': np.array([.7, .3]),
        'strong_minority': np.array([.9, .1])},
       'outcomes': {
           'balanced': np.array([[.333, .333, .334],
                                [.333, .333, .334]]),
           'one_rare': np.array([[.1, .5, .4],
                                 [.1, .5, .4]]),
           'two_rare': np.array([[.1, .8, .1],
                                 [.1, .8, .1]]),
           },
       'bias': {
           'low': [np.array([[.9, .05, .05],
                             [.05, .9, .05],
                             [.05, .05, .9]]),
                   np.array([[.8, .1, .1],
                             [.1, .8, .1],
                             [.1, .1, .8]])],
           'medium': [np.array([[.9, .05, .05],
                                   [.05, .9, .05],
                                   [.05, .05, .9]]),
                        np.array([[.7, .15, .15],
                                  [.15, .7, .15],
                                  [.15, .15, .7]])],
           'high': [np.array([[.9, .05, .05],
                                   [.05, .9, .05],
                                   [.05, .05, .9]]),
                         np.array([[.5, .25, .25],
                                   [.25, .5, .25],
                                   [.25, .25, .5]])]
           }
       }

input_23 = [[[(p23['groups'][g], p23['outcomes'][o], p23['bias'][b]) 
        for b in bias_types] 
       for o in out_types] 
      for g in group_types]
input_23 = tools.flatten([l for l in tools.flatten(input_23)])

# Running the sim
with Pool() as p:
    input = [[[(outcomes, 
                groups2, 
                t[0], 
                t[1], 
                t[2], 
                loss, 
                goal,
                sits[i][0],
                sits[i][1],
                sits[i][2])
            for i, t in enumerate(input_23)]
           for loss in losses]
          for goal in goals]
    input = tools.flatten([l for l in tools.flatten(input)])
    res = p.starmap(tools.test_run, input)
    p.close()
    p.join()

stats_23 = pd.concat([r['stats'] for r in res], axis=0)
stats_23.to_csv('2-group 3-class stats.csv', index=False)

# Setting up a 3-group 3-group problem
bias_types = ['low_one', 'medium_one', 'high_one',
              'low_two', 'medium_two', 'high_two']
out_types = ['balanced', 'one_rare', 'two_rare']
group_types = ['no_minority', 'one_slight_minority', 'one_strong_minority',
               'two_slight_minorities', 'two_strong_minorities']

sits = [[[[g, o, p] for p in bias_types]
             for o in out_types]
            for g in group_types]
sits = tools.flatten([l for l in tools.flatten(sits)])

p33 = {'groups': {
        'no_minority': np.array([.33, .33, .34]),
        'one_slight_minority': np.array([.4, .4, .2]),
        'one_strong_minority': np.array([.45, .45, .1]),
        'two_slight_minorities': np.array([.6, .2, .2]),
        'two_strong_minorities': np.array([.8, .1, .1])},
       'outcomes': {
           'balanced': np.array([[.333, .333, .334],
                                [.333, .333, .334],
                                [.333, .333, .334]]),
           'one_rare': np.array([[.1, .5, .4],
                                 [.1, .5, .4],
                                 [.1, .5, .4]]),
           'two_rare': np.array([[.1, .8, .1],
                                 [.1, .8, .1],
                                 [.1, .8, .1]]),
           },
       'bias': {
           'low_one': [np.array([[.9, .05, .05],
                                 [.05, .9, .05],
                                 [.05, .05, .9]]),
                       np.array([[.9, .05, .05],
                                 [.05, .9, .05],
                                 [.05, .05, .9]]),
                        np.array([[.8, .1, .1],
                                  [.1, .8, .1],
                                  [.1, .1, .8]])],
           'medium_one': [np.array([[.9, .05, .05],
                                   [.05, .9, .05],
                                   [.05, .05, .9]]),
                          np.array([[.9, .05, .05],
                                    [.05, .9, .05],
                                    [.05, .05, .9]]),
                        np.array([[.7, .15, .15],
                                  [.15, .7, .15],
                                  [.15, .15, .7]])],
           'high_one': [np.array([[.9, .05, .05],
                                   [.05, .9, .05],
                                   [.05, .05, .9]]),
                        np.array([[.9, .05, .05],
                                  [.05, .9, .05],
                                  [.05, .05, .9]]),
                         np.array([[.5, .25, .25],
                                   [.25, .5, .25],
                                   [.25, .25, .5]])],
             'low_two': [np.array([[.9, .05, .05],
                                   [.05, .9, .05],
                                   [.05, .05, .9]]),
                         np.array([[.8, .1, .1],
                                   [.1, .8, .1],
                                   [.1, .1, .8]]),
                          np.array([[.8, .1, .1],
                                    [.1, .8, .1],
                                    [.1, .1, .8]])],
             'medium_two': [np.array([[.9, .05, .05],
                                     [.05, .9, .05],
                                     [.05, .05, .9]]),
                            np.array([[.7, .15, .15],
                                      [.15, .7, .15],
                                      [.15, .15, .7]]),
                          np.array([[.7, .15, .15],
                                    [.15, .7, .15],
                                    [.15, .15, .7]])],
             'high_two': [np.array([[.9, .05, .05],
                                     [.05, .9, .05],
                                     [.05, .05, .9]]),
                          np.array([[.5, .25, .25],
                                    [.25, .5, .25],
                                    [.25, .25, .5]]),
                           np.array([[.5, .25, .25],
                                     [.25, .5, .25],
                                     [.25, .25, .5]])]
           }
       }

input_33 = [[[(p33['groups'][g], p33['outcomes'][o], p33['bias'][b]) 
        for b in bias_types] 
       for o in out_types] 
      for g in group_types]
input_33 = tools.flatten([l for l in tools.flatten(input_33)])

# Running the sim
with Pool() as p:
    input = [[[(outcomes, 
                groups3, 
                t[0], 
                t[1], 
                t[2], 
                loss, 
                goal,
                sits[i][0],
                sits[i][1],
                sits[i][2])
            for i, t in enumerate(input_33)]
           for loss in losses]
          for goal in goals]
    input = tools.flatten([l for l in tools.flatten(input)])
    res = p.starmap(tools.test_run, input)
    p.close()
    p.join()

stats_33 = pd.concat([r['stats'] for r in res], axis=0)
stats_33.to_csv('3-group 3-class stats.csv', index=False)

