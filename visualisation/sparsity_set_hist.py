import matplotlib.pyplot as plt
import numpy as np

# Titles for each histogram
hist_titles = [
    '[Og] Amazon', '[Og] Google', '[Og] Yelp',
    '[Re] Amazon', '[Re] Google', '[Re] Yelp'
]

# Names of the sparsity sets (top axis labels)
sparsity_names = ['zero‑shot', 'tst1', 'tst2', 'tst3', 'tst4', 'tst5']

# Exact interval strings for bottom axis, for each dataset
ranges_str = [
    ["[0, 0]", "[1, 9]", "(9, 20]", "(20, 42]", "(42, 95]", "(95, 548]"],   # Og Amazon
    ["[0, 0]", "[1, 4]", "(4, 7]", "(7, 10]", "(10, 17]", "(17, 113]"],    # Og Google
    ["[0, 0]", "[1, 4]", "(4, 8]", "(8, 14]", "(14, 28]", "(28, 262]"],    # Og Yelp
    ["[0, 0]", "[1, 4]", "(4, 6]", "(6, 11]", "(11, 28]", "(28, 251]"],    # Re Amazon
    ["[0, 0]", "[1, 3]", "(3, 6]", "(6, 9]", "(9, 16]", "(16, 110]"],     # Re Google
    ["[0, 0]", "[1, 3]", "(3, 5]", "(5, 8]", "(8, 15]", "(15, 134]"]      # Re Yelp
]

# Percentages for each bin, for each dataset
percentages = [
    [12, 17, 18, 17, 18, 18],   # Og Amazon
    [ 9, 20, 19, 16, 18, 18],   # Og Google
    [10, 19, 20, 16, 17, 18],   # Og Yelp
    [ 2, 27, 13, 19, 20, 19],   # Re Amazon
    [ 2, 21, 26, 16, 17, 18],   # Re Google
    [ 4, 28, 17, 15, 17, 19]    # Re Yelp
]

fig, axes = plt.subplots(1, 6, figsize=(24, 4), sharey=True)

for i, ax in enumerate(axes):
    x = np.arange(1, 7)
    bars = ax.bar(x, percentages[i], width=0.6, align='center')
    
    # Add percentage text above each bar
    for bar, pct in zip(bars, percentages[i]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f'{pct}%',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    # Title and shared y-label
    ax.set_title(hist_titles[i])
    if i == 0:
        ax.set_ylabel('Percentage of Test Set')
    ax.set_ylim([0, 35])
    
    # Bottom x-axis: interaction ranges
    ax.set_xticks(x)
    ax.set_xticklabels(ranges_str[i], rotation=45, ha='right')
    ax.set_xlabel("Interaction range")
    
    # Top x-axis: sparsity set names
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(sparsity_names, rotation=45, ha='left')
    ax_top.xaxis.set_ticks_position('top')
    ax_top.xaxis.set_label_position('top')

plt.tight_layout()
plt.show()

output_path = './visualisation/sparsity_set_histograms.pdf'
fig.savefig(output_path, format='pdf')