use crate::{
    layer::Layer,
    network::Network,
    neuron::Neuron,
    types::{Dataset, NetworkConfig},
};
use rand::random;

#[derive(Debug, Clone)]
pub struct Batch {
    pub networks: Vec<Network>,
}

impl Batch {
    pub fn new_with_networks(networks: Vec<Network>) -> Self {
        Self { networks }
    }

    pub fn new_with_population(batch_size: u64, network_config: NetworkConfig) -> Self {
        Self {
            networks: (0..batch_size)
                .map(|_| create_random_network(network_config.clone()))
                .collect::<Vec<Network>>(),
        }
    }

    /// Evaluates the fitness of a batch of networks
    /// # Returns
    /// A sorted vector of networks with the best fitness
    pub fn eval_fitness(&self, training_data: Dataset) -> Vec<(f32, Network)> {
        let mut top_networks: Vec<(f32, Network)> = Vec::new();

        for network in &self.networks {
            // Evaluate the fitness of the network
            let mut fitness = 0.0;

            for (inputs, expected) in training_data.iter() {
                let actual = network.forward(inputs.clone());

                fitness += cost(expected.to_vec(), actual);
            }

            // Average the fitness
            fitness /= training_data.len() as f32;

            // Insert the network into the correct position using binary search
            let insert_pos =
                match top_networks.binary_search_by(|(f, _)| f.partial_cmp(&fitness).unwrap()) {
                    Ok(pos) => pos,
                    Err(pos) => pos,
                };

            top_networks.insert(insert_pos, (fitness, network.clone()));
        }

        top_networks
    }
}

fn cost(expected: Vec<f32>, actual: Vec<f32>) -> f32 {
    expected
        .iter()
        .zip(actual.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum()
}

/// Creates a random network with the given configuration
pub fn create_random_network(network_config: NetworkConfig) -> Network {
    let (input_size, hidden_layer_sizes, output_size) = network_config;

    // Keep track of the previous input size
    let mut previous_input_size = input_size;

    // Create the hidden layers
    let hidden_layers = hidden_layer_sizes
        .iter()
        .map(|size| {
            // Create the layer with the previous input size
            let layer = create_random_layer(previous_input_size, *size);

            // Update the previous input size
            previous_input_size = *size;

            // Return the layer
            layer
        })
        .collect::<Vec<Layer>>();

    // Create the output layer
    let output_layer = create_random_layer(previous_input_size, output_size);

    // Create the network
    Network::new(hidden_layers, output_layer)
}

/// Creates a random layer with the given size and input size
fn create_random_layer(input_size: u64, size: u64) -> Layer {
    Layer::new(
        (0..size)
            .map(|_| create_random_neuron(input_size))
            .collect(),
    )
}

/// Creates a random neuron with the given input size
fn create_random_neuron(input_size: u64) -> Neuron {
    Neuron::new(
        (0..input_size).map(|_| random::<f32>()).collect(),
        random::<f32>(),
    )
}
