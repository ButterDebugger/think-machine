pub type Inputs = Vec<f32>;
pub type Outputs = Vec<f32>;
pub type Dataset = Vec<(Inputs, Outputs)>;

#[derive(Debug, Clone)]
pub struct NetworkConfig {
    pub input_size: u64,
    pub hidden_layer_sizes: Vec<u64>,
    pub output_size: u64,
}
