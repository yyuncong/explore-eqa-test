import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# # 1. the width of the plot should be lower
# # 2. the hline should be vline
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.tight_layout()
# # Data for goatbench
# n_gb = [1, 3, 5, 7]
# gb_sr = [68.71, 68.71, 67.99, 66.55]
# gb_spl = [45.25, 48.43, 48.58, 48.4]
#
# # Create a figure and axes
# fig, ax1 = plt.subplots(figsize=(8, 6))
# ax2 = ax1.twinx()
#
# # Plot goatbench data
# sns.lineplot(x=n_gb, y=gb_sr,
#         marker='o', linestyle='-', color='blue', label='Success Rate',
#         linewidth=2, ax=ax1)
# sns.lineplot(x=n_gb, y=gb_spl,
#         marker='o', linestyle='-', color='red', label='SPL',
#         linewidth=2, ax=ax2)
#
# # Customize the plot
# ax1.set_xlabel('Number of Observations at Each Step', fontsize = 20)
# ax1.set_ylabel('Success Rate', fontsize = 20, color='blue')
# ax1.set_title('GOAT-Bench', fontsize = 20)
# ax1.legend(loc="upper left", fontsize = 14)
# ax1.set_ylim([60, 75])
# ax1.grid(True)
#
# ax2.set_ylabel('SPL', fontsize = 20, color='red')
# ax2.set_ylim([44, 50])
# ax2.legend(loc="upper right", fontsize = 14)
#
# plt.savefig('plot_n_goatbench.png', dpi = 200)



# # 1. the width of the plot should be lower
# # 2. the hline should be vline
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.tight_layout()
#
# # Data for aeqa
# n_aeqa = [1, 3, 5, 7]
# aeqa_sr = [49.86, 52.58, 52.17, 50.68]
# aeqa_spl = [37.29, 42, 41.51	, 40.26]
#
# # Create a figure and axes
# fig, ax1 = plt.subplots(figsize=(8, 6))
# ax2 = ax1.twinx()
#
# # Plot goatbench data
# sns.lineplot(x=n_aeqa, y=aeqa_sr,
#         marker='o', linestyle='-', color='blue', label='LLM-Match',
#         linewidth=2, ax=ax1)
# sns.lineplot(x=n_aeqa, y=aeqa_spl,
#         marker='o', linestyle='-', color='red', label='LLM-Match SPL',
#         linewidth=2, ax=ax2)
#
# # Customize the plot
# ax1.set_xlabel('Number of Observations at Each Step', fontsize = 20)
# ax1.set_ylabel('LLM-Match', fontsize = 20, color='blue')
# ax1.set_title('A-EQA', fontsize = 20)
# ax1.legend(loc="upper left", fontsize = 14)
# ax1.set_ylim([45, 55])
# ax1.grid(True)
#
# ax2.set_ylabel('LLM-Match SPL', fontsize = 20, color='red')
# ax2.set_ylim([35, 45])
# ax2.legend(loc="upper right", fontsize = 14)
#
# plt.savefig('plot_n_aeqa.png', dpi = 200)



# # min_dist
# # Data for goatbench
# min_dist = [1.5, 2.5, 3.5, 4.75, 6]
# gb_sr = [55.4, 63.67, 68.71, 70.5, 70.86]
# gb_spl = [39.41, 42.35, 48.43, 51.48, 48.12]
#
# # Set font to Times New Roman and apply tight layout
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.tight_layout()
#
# # Create a figure and axes
# fig, ax1 = plt.subplots(figsize=(8, 6))
#
# # Plot success rate and SPL on the primary y-axis using sns.lineplot
# sns.lineplot(x=min_dist, y=gb_sr, marker='o', linestyle='-', color='blue', label='Success Rate', linewidth=2, ax=ax1)
#
# # Customize ax1
# ax1.set_ylabel('Success Rate', fontsize=20, color='blue')
# ax1.set_xlabel('Maximum Distance', fontsize=20)
# ax1.set_title("GOAT-Bench", fontsize=20)
# ax1.grid(True)
# ax1.set_ylim([50, 75])
# ax1.set_xlim([1, 7])
#
# ax2 = ax1.twinx()
# sns.lineplot(x=min_dist, y=gb_spl, marker='o', linestyle='-', color='red', label='SPL', linewidth=2, ax=ax2)
# ax2.set_ylim([37, 56])
# ax2.set_ylabel('SPL', fontsize=20, color='red')
#
# # Add legends for both axes
# ax1.legend(loc="upper left", fontsize=14)
# ax2.legend(loc="upper right", fontsize=14)
#
# # Save the figure
# plt.savefig('plot_min_dist_goatbench.png', dpi=200)

# # Data for aeqa
# min_dist = [1.5, 2.5, 3.5, 4.75, 6]
# aeqa_sr = [53.53, 54.76, 52.58, 51.36, 47.83]
# aeqa_spl = [38.25, 39.74, 42, 40.23, 37.31]
#
# # Set font to Times New Roman and apply tight layout
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.tight_layout()
#
# # Create a figure and axes
# fig, ax1 = plt.subplots(figsize=(8, 6))
#
# # Plot success rate and SPL on the primary y-axis using sns.lineplot
# sns.lineplot(x=min_dist, y=aeqa_sr, marker='o', linestyle='-', color='blue', label='LLM-Match', linewidth=2, ax=ax1)
#
# # Customize ax1
# ax1.set_ylabel('LLM-Match', fontsize=20, color='blue')
# ax1.set_xlabel('Maximum Distance', fontsize=20)
# ax1.set_title("A-EQA", fontsize=20)
# ax1.grid(True)
# ax1.set_ylim([44, 57])
# ax1.set_xlim([1, 7])
#
# ax2 = ax1.twinx()
# sns.lineplot(x=min_dist, y=aeqa_spl, marker='o', linestyle='-', color='red', label='LLM-Match SPL', linewidth=2, ax=ax2)
# ax2.set_ylim([36, 46])
# ax2.set_ylabel('LLM-Match SPL', fontsize=20, color='red')
#
# # Add legends for both axes
# ax1.legend(loc="upper left", fontsize=14)
# ax2.legend(loc="upper right", fontsize=14)
#
# # Save the figure
# plt.savefig('plot_min_dist_aeqa.png', dpi=200)




# # K
# Data for goatbench
K = [1, 2, 3, 5, 10]
gb_sr = [64.03, 67.63, 67.63, 68.35, 69.1]
gb_spl = [45.9, 48.07, 47.71, 48.17, 48.9]
filtered_over_all = np.asarray([1.77, 2.71, 3.61, 4.39, 4.66]) / np.asarray([18.44, 17.39, 17.75, 16.94, 16.6])

# Set font to Times New Roman and apply tight layout
plt.rcParams['font.family'] = 'Times New Roman'
plt.tight_layout()

# Create a figure and axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot success rate and SPL on the primary y-axis using sns.lineplot
sns.lineplot(x=K, y=gb_sr, marker='o', linestyle='-', color='blue', label='Success Rate', linewidth=2, ax=ax1)

# Customize ax1
ax1.set_ylabel('Success Rate', fontsize=20, color='blue')
ax1.set_xlabel('Number of Filtered Object Classes', fontsize=20)
ax1.set_title("GOAT-Bench", fontsize=20)
ax1.grid(True)
ax1.set_ylim([58, 71])
ax1.set_xlim([0, 11])

ax2 = ax1.twinx()
sns.lineplot(x=K, y=gb_spl, marker='o', linestyle='-', color='red', label='SPL', linewidth=2, ax=ax2)
ax2.set_ylim([44, 50])
ax2.set_ylabel('SPL', fontsize=20, color='red')

# Create a secondary y-axis for filtered snapshot ratio
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
sns.lineplot(x=K, y=filtered_over_all, marker='o', linestyle='--', color='green', label='Filtered Snapshot Ratio', linewidth=2, ax=ax3)
ax3.set_ylabel('Filtered Snapshot Ratio', fontsize=20, color='green')
ax3.set_ylim([0, 0.4])

# Add legends for both axes
ax1.legend(loc="upper left", fontsize=14)
ax2.legend(loc="upper center", fontsize=14)
ax3.legend(loc="upper right", fontsize=14)

# Adjust layout to give space for ax3's label
plt.subplots_adjust(left=0.1, right=0.85)  # Increase right margin

# Save the figure
plt.savefig('plot_k_goatbench.png', dpi=200)


# # data for aeqa
# K = [1, 2, 3, 5, 10]
# aeqa_sr = [49.48, 49.59, 51.09, 52.04, 52.6]
# aeqa_spl = [38.01, 38.1, 39.58, 40.87, 42]
# filtered_over_all = np.asarray([1.4, 2.26, 2.7, 3.15, 3.26]) / np.asarray([11.02, 10.83, 10.55, 10.65, 10.93])
#
# # Set font to Times New Roman and apply tight layout
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.tight_layout()
#
# # Create a figure and axes
# fig, ax1 = plt.subplots(figsize=(10, 6))
#
# # Plot success rate and SPL on the primary y-axis using sns.lineplot
# sns.lineplot(x=K, y=aeqa_sr, marker='o', linestyle='-', color='blue', label='LLM-Match', linewidth=2, ax=ax1)
#
# # Customize ax1
# ax1.set_ylabel('LLM-Match', fontsize=20, color='blue')
# ax1.set_xlabel('Number of Filtered Object Classes', fontsize=20)
# ax1.set_title("A-EQA", fontsize=20)
# ax1.grid(True)
# ax1.set_ylim([48, 54])
# ax1.set_xlim([0, 11])
#
# ax2 = ax1.twinx()
# sns.lineplot(x=K, y=aeqa_spl, marker='o', linestyle='-', color='red', label='LLM-Match SPL', linewidth=2, ax=ax2)
# ax2.set_ylim([37, 44])
# ax2.set_ylabel('LLM-Match SPL', fontsize=20, color='red')
#
# # Create a secondary y-axis for filtered snapshot ratio
# ax3 = ax1.twinx()
# ax3.spines['right'].set_position(('outward', 60))
#
#
# sns.lineplot(x=K, y=filtered_over_all, marker='o', linestyle='--', color='green', label='Filtered Snapshot Ratio', linewidth=2, ax=ax3)
# ax3.set_ylabel('Filtered Snapshot Ratio', fontsize=20, color='green')
# ax3.set_ylim([0, 0.4])
#
# # Add legends for both axes
# ax1.legend(loc="upper left", fontsize=14)
# ax2.legend(loc="upper center", fontsize=14)
# ax3.legend(loc="upper right", fontsize=14)
#
# # Adjust layout to give space for ax3's label
# plt.subplots_adjust(left=0.1, right=0.85)  # Increase right margin
#
# # Save the figure
# plt.savefig('plot_k_aeqa.png', dpi=200)
