use crate::{
    neuron::{Neuron, sigmoid_derivative},
    types::{Inputs, Outputs},
};

#[derive(Debug, Clone)]
pub struct Layer {
    neurons: Vec<Neuron>,
    input_size: u64,
}

impl Layer {
    pub fn new(input_size: u64, neurons: Vec<Neuron>) -> Self {
        Self {
            neurons,
            input_size,
        }
    }

    /// Creates a random layer with the given size and input size
    pub fn new_with_random_values(input_size: u64, size: u64) -> Self {
        Self {
            neurons: (0..size)
                .map(|_| Neuron::new_with_random_values(input_size))
                .collect(),
            input_size,
        }
    }

    pub fn forward(&mut self, inputs: Inputs) -> Outputs {
        self.neurons
            .iter_mut()
            .map(|neuron| neuron.forward(inputs.clone()))
            .collect()
    }

    // pub fn mutate(&mut self, learning_rate: f32) -> &mut Self {
    //     // Mutate each neuron
    //     self.neurons.iter_mut().for_each(|neuron| {
    //         neuron.mutate(learning_rate);
    //     });

    //     // Return the layer
    //     self
    // }

    /// Backpropagate errors through the layer and returns the errors for the previous layer
    pub fn backward(
        &mut self,
        inputs: &Inputs,
        output_errors: &Outputs,
        learning_rate: f32,
    ) -> Inputs {
        let mut input_errors = vec![0.0; inputs.len()];

        // For each neuron in this layer
        for (neuron, &output_error) in self.neurons.iter_mut().zip(output_errors.iter()) {
            // Calculate the gradient for this neuron
            let activation_derivative = sigmoid_derivative(neuron.weight_sum);
            let gradient = output_error * activation_derivative;

            // Accumulate errors for the previous layer
            for (i, &weight) in neuron.weights.iter().enumerate() {
                input_errors[i] += gradient * weight;
            }

            // Update this neuron's weights and bias
            neuron.update_weights(inputs, gradient, learning_rate);
        }

        input_errors
    }
}
