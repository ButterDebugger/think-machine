use model::{network::Network, types::NetworkConfig};

#[derive(Debug, Clone)]
pub struct Batch {
    pub networks: Vec<Network>,
}

impl Batch {
    pub fn with_networks(networks: Vec<Network>) -> Self {
        Self { networks }
    }

    pub fn with_population(batch_size: u64, network_config: NetworkConfig) -> Self {
        Self {
            networks: (0..batch_size)
                .map(|_| Network::new_with_random_values(network_config.clone()))
                .collect::<Vec<Network>>(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct FittedBatch {
    top_networks: Vec<(f32, Network)>,
}

impl FittedBatch {
    pub fn add_network(&mut self, fitness: f32, network: Network) {
        // Insert the network into the correct position using binary search
        let insert_pos = match self
            .top_networks
            .binary_search_by(|(f, _)| f.partial_cmp(&fitness).unwrap())
        {
            Ok(pos) => pos,
            Err(pos) => pos,
        };

        self.top_networks
            .insert(insert_pos, (fitness, network.clone()));
    }

    pub fn get_best_network(&self) -> Option<Network> {
        self.top_networks
            .first()
            .map(|(_, network)| network.clone())
    }

    pub fn get_best_fitness(&self) -> Option<f32> {
        self.top_networks.first().map(|(fitness, _)| *fitness)
    }

    pub fn get_top_networks(&self) -> &Vec<(f32, Network)> {
        &self.top_networks
    }
}

// impl From<FittedBatch> for Batch {
//     fn from(val: FittedBatch) -> Self {
//         Batch {
//             networks: val
//                 .top_networks
//                 .into_iter()
//                 .map(|(_, network)| network)
//                 .collect(),
//         }
//     }
// }
