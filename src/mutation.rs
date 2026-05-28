use model::{layer::Layer, network::Network, neuron::Neuron, types::Dataset};
use training::{
    batch::{Batch, FittedBatch},
    trainer::Trainable,
};

pub struct Mutation {
    learning_rate: f32,
    training_data: Dataset,
}

impl Mutation {
    pub fn new(learning_rate: f32, training_data: Dataset) -> Self {
        Self {
            learning_rate,
            training_data,
        }
    }
}

impl Trainable for Mutation {
    fn step(&mut self, batch: &mut Batch) {
        // Calculate the top networks in the batch
        let fitted_batch = self.eval_batch_fitness(batch);
        let top_networks = fitted_batch.get_top_networks();

        // Calculate the number of networks to keep and mutate
        let batch_size = batch.networks.len() as u64;
        let top_count = (batch_size / 2) as usize;
        let mutate_count = batch_size as usize - top_count;

        // Keep the top networks
        let mut new_networks: Vec<Network> = top_networks
            .iter()
            .take(top_count)
            .map(|(_, network)| network.clone())
            .collect();

        // Mutate copies of the top networks
        new_networks.extend(top_networks.iter().take(mutate_count).map(|(_, network)| {
            let mut cloned = network.clone();
            mutate_network(&mut cloned, self.learning_rate);
            cloned
        }));

        // Update the batch with the new networks
        batch.networks = new_networks;
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

fn mutate_network(network: &mut Network, learning_rate: f32) {
    // Mutate the hidden layers
    network.hidden_layers.iter_mut().for_each(|layer| {
        mutate_layer(layer, learning_rate);
    });

    // Mutate the output layer
    mutate_layer(&mut network.output_layer, learning_rate);
}

fn mutate_layer(layer: &mut Layer, learning_rate: f32) {
    // Mutate each neuron
    layer.neurons.iter_mut().for_each(|neuron| {
        mutate_neuron(neuron, learning_rate);
    });
}

/// Mutates the neuron with random weights and bias
fn mutate_neuron(neuron: &mut Neuron, learning_rate: f32) {
    // Mutate the weights
    neuron.weights.iter_mut().for_each(|weight| {
        *weight += (rand::random::<f32>() - 0.5) * learning_rate;
    });

    // Mutate the bias
    neuron.bias += (rand::random::<f32>() - 0.5) * learning_rate;
}

fn cost(expected: Vec<f32>, actual: Vec<f32>) -> f32 {
    expected
        .iter()
        .zip(actual.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
}
