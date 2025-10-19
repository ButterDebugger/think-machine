use crate::{
    layer::Layer,
    types::{Inputs, Outputs},
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
