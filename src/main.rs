use crate::{layer::Layer, network::Network, neuron::Neuron};
use std::vec;

mod layer;
mod network;
mod neuron;

fn main() {
    let mut network = Network::new(Layer::new(vec![
        Neuron::new(vec![1.0, 1.0], 0.0),
        Neuron::new(vec![1.0, 1.0], 0.0),
    ]));

    network.add_hidden_layer(Layer::new(vec![
        Neuron::new(vec![1.0, 1.0], 0.0),
        Neuron::new(vec![1.0, 1.0], 0.0),
    ]));

    let output = network.forward(vec![0.0, 0.0]);

    println!("Hello, world! {:#?}", output);
}
