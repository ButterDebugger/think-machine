/// Sigmoid activation function
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Derivative of the sigmoid activation function
pub fn sigmoid_derivative(x: f32) -> f32 {
    let activation = sigmoid(x);
    activation * (1.0 - activation)
}
