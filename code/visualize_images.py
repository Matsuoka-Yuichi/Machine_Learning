import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_test_images(network, test_data, num_images=10, save_path='../data/visualized_images'):
    # Create the directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Determine the grid size
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    # Create a figure for the grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for i in range(num_images):
        image = test_data[i][0].reshape(28, 28)  # Assuming the images are 28x28 pixels
        label = test_data[i][1]
        prediction = network.feedforward(test_data[i][0])
        predicted_label = np.argmax(prediction)

        ax = axes[i // grid_size, i % grid_size]
        ax.imshow(image, cmap='gray')
        ax.set_title(f"True: {label}\nPred: {predicted_label}")
        ax.axis('off')

    # Hide any unused subplots
    for j in range(num_images, grid_size * grid_size):
        fig.delaxes(axes.flatten()[j])

    # Save the figure
    save_file = os.path.join(save_path, 'visualized_images.png')
    plt.savefig(save_file)
    print(f"Saved visualized images to {save_file}")
    plt.close(fig)


