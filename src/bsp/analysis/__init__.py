import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 8,       # Base font size for axes labels
    'axes.titlesize': 8,  # Font size for axes titles
    'axes.labelsize': 8,  # Font size for axes labels
    'xtick.labelsize': 6,  # Font size for x-tick labels
    'ytick.labelsize': 6,  # Font size for y-tick labels
    'legend.fontsize': 6,  # Font size for the legend
})
# Ensure text remains editable and fonts are embedded
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'