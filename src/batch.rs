use crate::{
    network::Network,
    types::{Dataset, NetworkConfig},
};

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
                .map(|_| Network::new_with_random_values(network_config.clone()))
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
