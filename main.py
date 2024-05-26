import os
import argparse
from code.network import Network, load_data, prepare_data
from code.visualize_images import visualize_all_test_images

def main(visualize_training, save_weights):
    # Load and prepare data
    training_data, validation_data, test_data = load_data()
    training_data, validation_data, test_data = prepare_data(training_data, validation_data, test_data)

    # Initialize the network with 784 input neurons, 30 hidden neurons, and 10 output neurons
    net = Network([784, 30, 10])


    # Train the network using stochastic gradient descent
    net.SGD(training_data, 10, 10, 0.3, test_data=test_data)
    # net.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data)

    # Evaluate the network on the test data
    accuracy = net.evaluate(test_data)
    print(f"Accuracy on test data: {accuracy} / {len(test_data)}")

        # Visualize all test images and save the visualization
    visualize_all_test_images(net, test_data, save_path='../data/visualized_images')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network and optionally visualize training images and save weights.")
    parser.add_argument('--visualize-training', action='store_true', help="Visualize training images")
    parser.add_argument('--save-weights', action='store_true', help="Save weights after training")
    args = parser.parse_args()

    main(args.visualize_training, args.save_weights)

