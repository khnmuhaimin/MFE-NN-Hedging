# MFE-NN-Hedging

# Things We Need to Decide On
1. What range of value should we use for sigma
2. What range for S/K
3. How many trading steps/what range of N (hopefully our model generalizes to different N values)
4. Should we input nothing about volatility, historical volatility, or true volatility
5. should we input the BSM ouptut as an input.
6. How should we generate stock price paths (how does each method affect convergence and performance)
7. How do we test our model
8. What do we report on
9. What loss function do we choose
10. What pricing function do we choose