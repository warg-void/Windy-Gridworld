# Windy Gridworld
This repository contains my solution to the windy gridworld problem from Sutton & Barto. There is a gridworld with a crosswind running upward through the middle of the grid. The actions are the standard four—up, down, right, and left. The goal is the reach the end position from the start.

 The strength of wind is given below each column. If you are at a position with wind 2, and you choose to move right, you will move 2 upwards and 1 right. Wind was randomly initialized and 1000 simulations were ran for each of the algorithms with different wind speeds each time.

## Algorithms used
On-policy SARSA and Q-learning were used. To compare the two, a plot of the episode length over number of episodes was done. The lower the episode length, the faster the agent reached the goal. I also experimented with a few features of C++20.

## To Build (dependency fetched automatically, no installation required)
```
cmake -B build
cmake --build build
./build/out
```
![plot1](/SARSA%20Q-learn.png)
![plot2](/episode%20graph.png)