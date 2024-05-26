import os
import argparse
import csv
from itertools import product
from code.network import Network, load_data, prepare_data
from code.visualize_images import visualize_all_test_images

def run_experiment(visualize_training, save_weights, input_neurons, hidden_neurons, output_neurons, epochs, mini_batch_size, learning_rate, results_writer):
    # Load and prepare data
    training_data, validation_data, test_data = load_data()
    training_data, validation_data, test_data = prepare_data(training_data, validation_data, test_data)

    # Initialize the network
    net = Network([input_neurons] + hidden_neurons + [output_neurons])

    # Train the network using stochastic gradient descent
    net.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data=test_data)

    # Evaluate the network on the test data
    accuracy = net.evaluate(test_data)
    print(f"Accuracy on test data: {accuracy} / {len(test_data)}")

    # Visualize all test images and save the visualization
    if visualize_training:
        visualize_all_test_images(net, test_data, save_path='../data/visualized_images')

    # Write results to CSV
    results_writer.writerow(['Input Neurons', input_neurons])
    results_writer.writerow(['Hidden Neurons', hidden_neurons])
    results_writer.writerow(['Output Neurons', output_neurons])
    results_writer.writerow(['Epochs', epochs])
    results_writer.writerow(['Mini Batch Size', mini_batch_size])
    results_writer.writerow(['Learning Rate', learning_rate])
    results_writer.writerow(['Accuracy', f"{accuracy} / {len(test_data)}"])
    results_writer.writerow([])  # Empty row for separation
    results_writer.writerow(['Epoch', 'Cost', 'Accuracy'])
    for epoch in range(epochs):
        results_writer.writerow([epoch + 1, net.epoch_costs[epoch], net.epoch_accuracies[epoch]])
    results_writer.writerow([])  # Empty row for separation

def main(visualize_training, save_weights):
    # Parameters
    input_neurons = 784
    hidden_neurons_list = [[30], [30, 30]]
    output_neurons = 10
    epochs_list = [10, 20]
    mini_batch_size_list = [10, 20]
    learning_rate_list = [0.1, 0.3]

    # Create results folder and file
    results_folder = 'results'
    os.makedirs(results_folder, exist_ok=True)
    filename = "accumulated_results.csv"
    filepath = os.path.join(results_folder, filename)

    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Parameter', 'Value'])

        # Iterate over all combinations of parameters
        for hidden_neurons, epochs, mini_batch_size, learning_rate in product(hidden_neurons_list, epochs_list, mini_batch_size_list, learning_rate_list):
            run_experiment(visualize_training, save_weights, input_neurons, hidden_neurons, output_neurons, epochs, mini_batch_size, learning_rate, writer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network and optionally visualize training images and save weights.")
    parser.add_argument('--visualize-training', action='store_true', help="Visualize training images")
    parser.add_argument('--save-weights', action='store_true', help="Save weights after training")
    args = parser.parse_args()

    main(args.visualize_training, args.save_weights)