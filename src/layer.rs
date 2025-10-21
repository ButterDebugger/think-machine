use crate::{
    neuron::Neuron,
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

    pub fn mutate(&mut self, learning_rate: f32) -> &mut Self {
        // Mutate each neuron
        self.neurons.iter_mut().for_each(|neuron| {
            neuron.mutate(learning_rate);
        });

        // Return the layer
        self
    }
}
