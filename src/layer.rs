use crate::{
    neuron::Neuron,
    types::{Inputs, Outputs},
};

#[derive(Debug, Clone)]
pub struct Layer(Vec<Neuron>);

impl Layer {
    pub fn new(neurons: Vec<Neuron>) -> Self {
        Self(neurons)
    }

    pub fn forward(&self, inputs: Inputs) -> Outputs {
        self.0
            .iter()
            .map(|neuron| neuron.forward(inputs.clone()))
            .collect()
    }

    pub fn mutate(&self, learning_rate: f32) -> Layer {
        let neurons = self
            .0
            .iter()
            .map(|neuron| neuron.mutate(learning_rate))
            .collect::<Vec<Neuron>>();

        Layer::new(neurons)
    }
}
