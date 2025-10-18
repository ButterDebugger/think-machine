use crate::neuron::Neuron;

pub(crate) struct Layer(Vec<Neuron>);

impl Layer {
    pub(crate) fn new(neurons: Vec<Neuron>) -> Self {
        Self(neurons)
    }

    pub(crate) fn forward(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.0
            .iter()
            .map(|neuron| neuron.forward(&inputs.clone()))
            .collect()
    }
}
