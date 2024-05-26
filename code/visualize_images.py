import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def visualize_all_test_images(network, test_data, save_path='../data/visualized_images'):
    print("start image generation")
    # Create the directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    num_images = len(test_data)
    grid_size = int(np.ceil(np.sqrt(num_images)))

    # Create a figure for the grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(50, 50))
    fig.subplots_adjust(hspace=2.4, wspace=2.4)

    for i in range(num_images):
        image = test_data[i][0].reshape(28, 28)  # Assuming the images are 28x28 pixels
        label = test_data[i][1]
        prediction = network.feedforward(test_data[i][0])
        predicted_label = np.argmax(prediction)

        ax = axes[i // grid_size, i % grid_size]
        ax.imshow(image, cmap='gray')
        ax.set_title(f"True: {label}\nPred: {predicted_label}",fontsize=6)
        ax.axis('off')

    # Hide any unused subplots
    for j in range(num_images, grid_size * grid_size):
        fig.delaxes(axes.flatten()[j])

    # Save the figure
    save_file = os.path.join(save_path, f'all_test_images_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png')
    plt.savefig(save_file)
    print(f"Saved visualized images to {save_file}")
    plt.close(fig)
    print("finished_image_generation")

