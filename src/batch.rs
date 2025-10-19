use crate::{layer::Layer, network::Network, neuron::Neuron, types::Dataset};
use rand::random;

#[derive(Debug, Clone)]
pub struct Batch {
    pub size: u64,
    pub networks: Vec<Network>,
}

impl Batch {
    pub fn new(size: u64) -> Self {
        Self {
            size,
            networks: Vec::new(),
        }
    }

    pub fn populate(&mut self, hidden_layer_sizes: Vec<u64>, output_size: u64) {
        for _ in self.networks.len()..(self.size as usize) {
            self.networks.push(create_random_network(
                hidden_layer_sizes.clone(),
                output_size,
            ));
        }
    }

    pub fn add_network(&mut self, network: Network) {
        self.networks.push(network);
    }

    /// Evaluates the fitness of a batch of networks
    /// # Returns
    /// A sorted vector of networks with the best fitness
    pub fn eval_fitness(self, training_data: Dataset) -> Vec<(f32, Network)> {
        let mut top_networks: Vec<(f32, Network)> = Vec::new();

        for network in self.networks {
            // Evaluate the fitness of the network
            let mut fitness = 0.0;

            for (inputs, expected) in training_data.iter() {
                let actual = network.forward(inputs.clone());

                fitness += cost(expected.to_vec(), actual);
            }

            // Average the fitness
            fitness /= training_data.len() as f32;

            // Insert the network into the correct position TODO: Binary search
            let mut i = 0;

            while i < top_networks.len() && top_networks[i].0 < fitness {
                i += 1;
            }

            top_networks.insert(i, (fitness, network));
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

pub fn create_random_network(hidden_layer_sizes: Vec<u64>, output_size: u64) -> Network {
    let hidden_layers = hidden_layer_sizes
        .iter()
        .map(|size| create_random_layer(*size))
        .collect::<Vec<Layer>>();

    let output_layer = create_random_layer(output_size);

    Network::new(hidden_layers, output_layer)
}
fn create_random_layer(size: u64) -> Layer {
    Layer::new((0..size).map(|_| create_random_neuron()).collect())
}

fn create_random_neuron() -> Neuron {
    Neuron::new(vec![random::<f32>(), random::<f32>()], random::<f32>())
}
