import os
import argparse
import csv
from code.network import Network, load_data, prepare_data
from code.visualize_images import visualize_all_test_images

def main(visualize_training):
    # Load and prepare data
    training_data, validation_data, test_data = load_data()
    training_data, validation_data, test_data = prepare_data(training_data, validation_data, test_data)

    # Parameters
    input_neurons = [784]
    hidden_neurons = [30,30]
    output_neurons = [10]
    epochs = 10
    mini_batch_size = 10
    learning_rate = 0.1

    # Initialize the network
    net = Network(input_neurons + hidden_neurons+ output_neurons)

    # Train the network using stochastic gradient descent
    net.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data=test_data)

    # Evaluate the network on the test data
    accuracy = net.evaluate(test_data)
    print(f"Accuracy on test data: {accuracy} / {len(test_data)}")


    # Visualize all test images and save the visualization
    if visualize_training:
        visualize_all_test_images(net, test_data, save_path='../data/visualized_images')

    # Save results to CSV
    results_folder = 'results'
    os.makedirs(results_folder, exist_ok=True)
    filename = f"results_epochs_{epochs}_batch_{mini_batch_size}_lr_{learning_rate}_hidden_{hidden_neurons}.csv"
    filepath = os.path.join(results_folder, filename)

    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Parameter', 'Value'])
        writer.writerow(['Input Neurons', input_neurons])
        writer.writerow(['Hidden Neurons', hidden_neurons])
        writer.writerow(['Output Neurons', output_neurons])
        writer.writerow(['Epochs', epochs])
        writer.writerow(['Mini Batch Size', mini_batch_size])
        writer.writerow(['Learning Rate', learning_rate])
        writer.writerow(['Accuracy', f"{accuracy} / {len(test_data)}"])
        writer.writerow([])  # Empty row for separation
        writer.writerow(['Epoch', 'Cost', 'Accuracy'])
        for epoch in range(epochs):
            writer.writerow([epoch + 1, net.epoch_costs[epoch], net.epoch_accuracies[epoch]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network and optionally visualize training images and save weights.")
    parser.add_argument('--visualize-training', action='store_true', help="Visualize training images")
    parser.add_argument('--save-weights', action='store_true', help="Save weights after training")
    args = parser.parse_args()

    main(args.visualize_training, args.save_weights)

