# ep-Explore
ep-Explore is a novel approach to training a chess engine using supervised learning on a single GPU and CPU.

The main objective of this algorithm is to train and optimize a neural network to play chess using limited memory, a single GPU (RTX Titan), and a single CPU (Threadripper). This algorithm attempts to bypass the reinforcement learning route (specifically AlphaZero's method of using MCTS to continually update policy and q-values) by utlizing an existing expert policy via Stockfish 10. This github post is meant to document how I design and continue to improve this algorithm.

ep-Explore's algorithm can be split into three phases: Sampling, Simulation, and Training. Its important to note that the first two phases, Sampling and Simulation, are independent of one another, meaning we will run them in parallel to decrease runtime by 100s of times. The algorithm's ability to label millions of positions in a few hours within these two phases seems to be the source of its performance!

# Sampling



# Simulation


# Training


# Improvements
