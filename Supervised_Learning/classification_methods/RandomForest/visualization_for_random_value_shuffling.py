import numpy as np
import matplotlib.pyplot as plt

# I made this code in order to fully understand how choosing different values for
# random_state parameter of train_test_split function would ffect the shuffling

# Define the dataset
data = np.array([1, 2, 3, 4, 5])

# Define the random_state values to visualize
random_states = [0, 1, 2, 3]

# Visualize the shuffling for each random_state value
fig, axs = plt.subplots(len(random_states), 1, figsize=(6, 8), sharex=True)
fig.suptitle('Visualization of Data Shuffling for Different random_state Values', fontsize=14)
for i, rs in enumerate(random_states):
    # Shuffle the dataset
    np.random.seed(rs)
    shuffled_data = np.random.permutation(data)
    # Plot the shuffled data
    axs[i].bar(range(len(data)), shuffled_data)
    axs[i].set_title(f'random_state = {rs}')
    axs[i].set_ylabel('Data')
    axs[i].set_ylim(0, 6)
plt.xlabel('Index')
plt.tight_layout()
plt.show()
