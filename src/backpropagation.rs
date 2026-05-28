use model::{
    layer::Layer,
    network::Network,
    neuron::{Neuron, sigmoid_derivative},
    types::{Dataset, Inputs, Outputs},
};
use training::{
    batch::{Batch, FittedBatch},
    trainer::Trainable,
};

pub struct Backpropagation {
    learning_rate: f32,
    training_data: Dataset,
}

impl Backpropagation {
    pub fn new(learning_rate: f32, training_data: Dataset) -> Self {
        Self {
            learning_rate,
            training_data,
        }
    }
}

impl Trainable for Backpropagation {
    fn step(&mut self, batch: &mut Batch) {
        // Backpropagate each network in the batch
        for network in &mut batch.networks {
            // Train the network on all the training data
            for (inputs, expected) in &self.training_data {
                backpropagate_network(
                    network,
                    inputs.clone(),
                    expected.clone(),
                    self.learning_rate,
                );
            }
        }
    }

    fn eval_batch_fitness(&self, batch: &Batch) -> FittedBatch {
        let mut fitted_batch = FittedBatch::default();

        for network in &batch.networks {
            let mut fitness = 0.0;
            let mut network = network.clone();

            for (inputs, expected) in self.training_data.iter() {
                let actual = network.forward(inputs.clone());

                fitness += cost(expected.to_vec(), actual);
            }

            // Average the fitness
            fitness /= self.training_data.len() as f32;

            // Add the network and its fitness to the fitted batch
            fitted_batch.add_network(fitness, network);
        }

        fitted_batch
    }
}

/// Backpropagate errors through the network and updates weights and biases
fn backpropagate_network(
    network: &mut Network,
    inputs: Inputs,
    expected: Outputs,
    learning_rate: f32,
) {
    // Forward pass and collect layer inputs
    let mut layer_inputs = vec![inputs.clone()];
    let mut current_input = inputs;

    for layer in &mut network.hidden_layers {
        current_input = layer.forward(current_input);
        layer_inputs.push(current_input.clone());
    }

    let outputs = network.output_layer.forward(current_input);
    let mut errors = cost_derivatives(expected, outputs);

    // Backpropagate through output layer
    errors = backpropagate_layer(
        &mut network.output_layer,
        &layer_inputs[network.hidden_layers.len()],
        &errors,
        learning_rate,
    );

    // Backpropagate through hidden layers in reverse
    for (i, layer) in network.hidden_layers.iter_mut().enumerate().rev() {
        errors = backpropagate_layer(layer, &layer_inputs[i], &errors, learning_rate);
    }
}

/// Backpropagate errors through the layer and returns the errors for the previous layer
fn backpropagate_layer(
    layer: &mut Layer,
    inputs: &Inputs,
    output_errors: &Outputs,
    learning_rate: f32,
) -> Inputs {
    let mut input_errors = vec![0.0; inputs.len()];

    // For each neuron in this layer
    for (neuron, &output_error) in layer.neurons.iter_mut().zip(output_errors.iter()) {
        // Calculate the gradient for this neuron
        let activation_derivative = sigmoid_derivative(neuron.weight_sum);
        let gradient = output_error * activation_derivative;

        // Accumulate errors for the previous layer
        for (i, &weight) in neuron.weights.iter().enumerate() {
            input_errors[i] += gradient * weight;
        }

        // Update this neuron's weights and bias
        update_neuron_weights(neuron, inputs, gradient, learning_rate);
    }

    input_errors
}

/// Updates the neuron's weights and bias based on gradients
fn update_neuron_weights(neuron: &mut Neuron, inputs: &Inputs, gradient: f32, learning_rate: f32) {
    // Update weights: w = w - learning_rate * gradient * input
    for (weight, input) in neuron.weights.iter_mut().zip(inputs.iter()) {
        *weight -= learning_rate * gradient * input;
    }

    // Update bias: b = b - learning_rate * gradient
    neuron.bias -= learning_rate * gradient;
}

fn cost(expected: Vec<f32>, actual: Vec<f32>) -> f32 {
    expected
        .iter()
        .zip(actual.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
}

fn cost_derivatives(expected: Vec<f32>, actual: Vec<f32>) -> Vec<f32> {
    expected
        .iter()
        .zip(actual.iter())
        .map(|(a, b)| 2.0 * (b - a))
        .collect::<Vec<f32>>()
}
