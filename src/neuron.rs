use crate::types::Inputs;
use rand::random;

#[derive(Debug, Clone)]
pub struct Neuron {
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl Neuron {
    pub fn new(weights: Vec<f32>, bias: f32) -> Self {
        Self { weights, bias }
    }

    /// Creates a random neuron with the given input size
    pub fn new_with_random_values(input_size: u64) -> Self {
        Self {
            weights: (0..input_size).map(|_| random::<f32>()).collect(),
            bias: random::<f32>(),
        }
    }

    pub fn forward(&self, inputs: Inputs) -> f32 {
        sigmoid(
            inputs
                .iter()
                .zip(self.weights.iter())
                .map(|(input, weight)| input * weight)
                .sum::<f32>()
                + self.bias,
        )
    }

    /// Creates a new neuron with random weights and bias
    pub fn mutate(&self, learning_rate: f32) -> Neuron {
        Neuron::new(
            self.weights
                .iter()
                .map(|weight| weight + (rand::random::<f32>() - 0.5) * learning_rate)
                .collect(),
            self.bias + (rand::random::<f32>() - 0.5) * learning_rate,
        )
    }
}

// Activation function
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
