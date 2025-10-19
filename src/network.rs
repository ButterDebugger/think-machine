use crate::{
    layer::Layer,
    types::{Inputs, NetworkConfig, Outputs},
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

    pub fn forward(&self, inputs: Inputs) -> Outputs {
        let outputs = self
            .hidden_layers
            .iter()
            .fold(inputs, |inputs, layer| layer.forward(inputs));

        self.output_layer.forward(outputs)
    }

    pub fn mutate(&self, learning_rate: f32) -> Network {
        let hidden_layers = self
            .hidden_layers
            .iter()
            .map(|layer| layer.mutate(learning_rate))
            .collect();

        let output_layer = self.output_layer.mutate(learning_rate);

        Network::new(hidden_layers, output_layer)
    }
}
