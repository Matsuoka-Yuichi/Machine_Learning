# Import necessary functions and classes from network.py
from code.network import Network, load_data, prepare_data
from code.visualize_images import visualize_test_images

# Load and prepare data
training_data, validation_data, test_data = load_data()
training_data, validation_data, test_data = prepare_data(training_data, validation_data, test_data)

# Initialize the network with 784 input neurons, 30 hidden neurons, and 10 output neurons
net = Network([784, 30, 10])
visualize_test_images(net, test_data, num_images=10)
# Train the network using stochastic gradient descent
net.SGD(training_data, 30, 10, 0.1, test_data=test_data)
# net.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data)


# Evaluate the network on the test data
accuracy = net.evaluate(test_data)
print(f"Accuracy on test data: {accuracy} / {len(test_data)}")