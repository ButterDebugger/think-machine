use crate::layer::Layer;

pub(crate) struct Network {
    pub(crate) hidden_layers: Vec<Layer>,
    pub(crate) output_layer: Layer,
}

impl Network {
    pub(crate) fn new(output_layer: Layer) -> Self {
        Self {
            hidden_layers: Vec::new(),
            output_layer,
        }
    }

    pub(crate) fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        let mut output = input;

        for layer in &mut self.hidden_layers {
            output = layer.forward(output);
        }

        self.output_layer.forward(output)
    }

    pub(crate) fn add_hidden_layer(&mut self, layer: Layer) {
        self.hidden_layers.push(layer);
    }
}
