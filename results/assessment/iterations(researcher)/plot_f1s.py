"""Plot F1 scores for all iterations. Iterations are on the X axis and F1 is on the Y axis. The plot is saved to results/assessment/iteration_scores.png."""

import pandas as pd
import matplotlib.pyplot as plt

results_df = pd.read_csv('results/assessment/iteration_scores.csv')
plt.plot(results_df['filename'], results_df['f1_macro'])
plt.xlabel('Iteration')
plt.ylabel('F1 Macro')
plt.title('F1 Macro Scores for Each Iteration')
plt.savefig('results/assessment/iteration_scores.png')