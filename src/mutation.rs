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
        todo!()
    }

    fn eval_batch_fitness(&self, batch: &Batch) -> FittedBatch {
        todo!()
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
