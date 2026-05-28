use crate::{backpropagation::Backpropagation, mutation::Mutation};
use model::types::NetworkConfig;
use std::vec;
use training::trainer::Trainer;

mod backpropagation;
mod mutation;

fn main() {
    let mut trainer = Trainer::new(
        100,
        NetworkConfig {
            input_size: 2,
            hidden_layer_sizes: vec![3, 4, 3],
            output_size: 8,
        },
        // Backpropagation::new(
        //     0.3,
        //     vec![
        //         (vec![0.0, 0.0], vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
        //         (vec![0.0, 1.0], vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        //         (vec![1.0, 0.0], vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
        //         (vec![1.0, 1.0], vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
        //     ],
        // ),
        Mutation::new(
            0.3,
            vec![
                (vec![0.0, 0.0], vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
                (vec![0.0, 1.0], vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
                (vec![1.0, 0.0], vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
                (vec![1.0, 1.0], vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
            ],
        ),
    );

    let fitted_batch = trainer.train(50, 100);
    let model = fitted_batch.get_best_network().unwrap();

    println!();
    println!("Model {:#?}", model);

    // println!("{:#?}", net.iter().map(|l| l.0).collect::<Vec<_>>());

    // Test the best network
    println!();
    println!("a=0 b=0 => {:#?}", model.clone().forward(vec![0.0, 0.0]));
    println!("a=0 b=1 => {:#?}", model.clone().forward(vec![0.0, 1.0]));
    println!("a=1 b=0 => {:#?}", model.clone().forward(vec![1.0, 0.0]));
    println!("a=1 b=1 => {:#?}", model.clone().forward(vec![1.0, 1.0]));
}
