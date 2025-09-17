import json
import matplotlib.pyplot as plt

# Your dictionary  #this is taken from final_output.txt
data = {
    "aruco10_0": {"y": 0.26153846153846155, "x": -0.7384615384615385},
    "aruco1_0": {"y": 1.1846153846153846, "x": -0.3769230769230769},
    "aruco2_0": {"y": 1.146153846153846, "x": -1.1153846153846154},
    "aruco3_0": {"y": -1.2461538461538462, "x": 1.123076923076923},
    "aruco4_0": {"y": 0.8, "x": 1.0153846153846153},
    "aruco5_0": {"y": 0.4846153846153846, "x": -0.03076923076923077},
    "aruco6_0": {"y": -0.07692307692307693, "x": 1.0384615384615385},
    "aruco7_0": {"y": -1.3846153846153846, "x": 0.24615384615384617},
    "aruco8_0": {"y": -0.8923076923076924, "x": -0.23076923076923078},
    "aruco9_0": {"y": -0.16153846153846155, "x": -1.2769230769230768},
    "tomato_0": {"y": -0.6461538461538462, "x": 0.9615384615384616},
    "capsicum_0": {"y": 0.4153846153846154, "x": 0.9538461538461539},
    "pear_0": {"y": -1.3384615384615384, "x": -0.6},
    "lemon_0": {"y": -0.7307692307692307, "x": -0.43846153846153846},
    "potato_0": {"y": -0.1, "x": -0.8923076923076924},
    "garlic_0": {"y": -0.25384615384615383, "x": -0.5538461538461539},
    "pumpkin_0": {"y": 0.9076923076923077, "x": -0.9615384615384616},
    "orange_0": {"y": 0.823076923076923, "x": -0.2076923076923077},
    "pear_1": {"y": 1.1307692307692307, "x": 0.8},
    "capsicum_1": {"y": -0.9923076923076923, "x": 0.5384615384615384}
}

# Create a scatter plot
fig, ax = plt.subplots(figsize=(6,6), facecolor='black')
ax.set_facecolor('black')
for name, coords in data.items():
    x, y = coords["x"], coords["y"]
    if "aruco" in name:
        plt.scatter(x, y, c="red", marker="s", label="Aruco" if "Aruco" not in plt.gca().get_legend_handles_labels()[1] else "")
    else:
        plt.scatter(x, y, c="blue", marker="o", label="Fruit" if "Fruit" not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.text(x+0.05, y+0.05, name, fontsize=8)

# Remove axes
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

plt.legend(facecolor='black', edgecolor='white', labelcolor='white')
plt.axis('equal')

# Save as PNG
plt.savefig("Week07-08/ground_truth_map.png", dpi=300, facecolor='black', bbox_inches='tight')
plt.show()