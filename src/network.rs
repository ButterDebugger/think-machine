use crate::{
    layer::Layer,
    types::{Dataset, Inputs, NetworkConfig, Outputs},
};

#[derive(Debug, Clone)]
pub struct Network {
    pub hidden_layers: Vec<Layer>,
    pub output_layer: Layer,
}

impl Network {
    pub fn new(hidden_layers: Vec<Layer>, output_layer: Layer) -> Self {
        Self {
            hidden_layers,
            output_layer,
        }
    }

    /// Creates a random network with the given configuration
    pub fn new_with_random_values(network_config: NetworkConfig) -> Self {
        let (input_size, hidden_layer_sizes, output_size) = network_config;

        // Keep track of the previous input size
        let mut previous_input_size = input_size;

        // Create the hidden layers
        let hidden_layers = hidden_layer_sizes
            .iter()
            .map(|size| {
                // Create the layer with the previous input size
                let layer = Layer::new_with_random_values(previous_input_size, *size);

                // Update the previous input size
                previous_input_size = *size;

                // Return the layer
                layer
            })
            .collect::<Vec<Layer>>();

        // Create the output layer
        let output_layer = Layer::new_with_random_values(previous_input_size, output_size);

        // Create the network
        Self {
            hidden_layers,
            output_layer,
        }
    }

    /// Calculates the fitness of the network
    pub fn fitness(&mut self, training_data: Dataset) -> f32 {
        let mut fitness = 0.0;

        for (inputs, expected) in training_data.iter() {
            let actual = self.forward(inputs.clone());

            fitness += cost(expected.to_vec(), actual);
        }

        // Return the average fitness
        fitness / training_data.len() as f32
    }

    pub fn forward(&mut self, inputs: Inputs) -> Outputs {
        let outputs = self
            .hidden_layers
            .iter_mut()
            .fold(inputs, |inputs, layer| layer.forward(inputs));

        self.output_layer.forward(outputs)
    }

    pub fn mutate(&mut self, learning_rate: f32) -> &mut Self {
        // Mutate the hidden layers
        self.hidden_layers.iter_mut().for_each(|layer| {
            layer.mutate(learning_rate);
        });

        // Mutate the output layer
        self.output_layer.mutate(learning_rate);

        // Return the network
        self
    }
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
        .map(|(a, b)| 2.0 * (a - b))
        .collect::<Vec<f32>>()
}
