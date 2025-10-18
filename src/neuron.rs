pub(crate) struct Neuron {
    pub(crate) weights: Vec<f32>,
    pub(crate) bias: f32,
}

impl Neuron {
    pub(crate) fn new(weights: Vec<f32>, bias: f32) -> Self {
        Self { weights, bias }
    }

    pub(crate) fn forward(&self, inputs: &[f32]) -> f32 {
        sigmoid(
            inputs
                .iter()
                .zip(self.weights.iter())
                .map(|(input, weight)| input * weight)
                .sum::<f32>()
                + self.bias,
        )
    }

    // pub(crate) fn forward(&self, inputs: &[f32]) -> f32 {
    //     let mut sum = 0.0;

    //     for (i, &weight) in self.weights.iter().enumerate() {
    //         sum += weight * inputs[i];
    //     }

    //     sum + self.bias
    // }
}

// Activation function
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
