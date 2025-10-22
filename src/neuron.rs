use crate::types::Inputs;
use rand::random;

#[derive(Debug, Clone)]
pub struct Neuron {
    pub weights: Vec<f32>,
    pub bias: f32,
    /// Cached weight sum
    pub weight_sum: f32,
    /// Cached activation value
    pub activation: f32,
}

impl Neuron {
    pub fn new(weights: Vec<f32>, bias: f32) -> Self {
        Self {
            weights,
            bias,
            weight_sum: 0.0,
            activation: 0.0,
        }
    }

    /// Creates a random neuron with the given input size
    pub fn new_with_random_values(input_size: u64) -> Self {
        Self {
            weights: (0..input_size).map(|_| random::<f32>()).collect(),
            bias: random::<f32>(),
            weight_sum: 0.0,
            activation: 0.0,
        }
    }

    pub fn forward(&mut self, inputs: Inputs) -> f32 {
        // Calculate the weight sum and cache it
        let weight_sum = inputs
            .iter()
            .zip(self.weights.iter())
            .map(|(input, weight)| input * weight)
            .sum::<f32>()
            + self.bias;

        self.weight_sum = weight_sum;

        // Calculate the activation value and cache it
        let activation = sigmoid(weight_sum);

        self.activation = activation;

        // Return the activation value
        activation
    }

    /// Mutates the neuron with random weights and bias
    pub fn mutate(&mut self, learning_rate: f32) -> &mut Self {
        // Mutate the weights
        self.weights.iter_mut().for_each(|weight| {
            *weight += (rand::random::<f32>() - 0.5) * learning_rate;
        });

        // Mutate the bias
        self.bias += (rand::random::<f32>() - 0.5) * learning_rate;

        // Return the neuron
        self
    }
}

/// Sigmoid activation function
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Derivative of the sigmoid activation function
fn sigmoid_derivative(x: f32) -> f32 {
    let activation = sigmoid(x);
    activation * (1.0 - activation)
}
