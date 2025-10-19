use crate::trainer::Trainer;
use std::vec;

mod batch;
mod layer;
mod network;
mod neuron;
mod trainer;
mod types;

fn main() {
    let mut trainer = Trainer::new(
        1000,
        0.3,
        vec![3, 4, 3],
        8,
        vec![
            (vec![0.0, 0.0], vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
            (vec![0.0, 1.0], vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
            (vec![1.0, 0.0], vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
            (vec![1.0, 1.0], vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
        ],
    );

    trainer.train(100, 100);

    let model = trainer.batch.networks[0].clone();

    // println!("{:#?}", net.iter().map(|l| l.0).collect::<Vec<_>>());

    // Test the best network
    println!();
    println!("a=0 b=0 => {:#?}", model.clone().forward(vec![0.0, 0.0]));
    println!("a=0 b=1 => {:#?}", model.clone().forward(vec![0.0, 1.0]));
    println!("a=1 b=0 => {:#?}", model.clone().forward(vec![1.0, 0.0]));
    println!("a=1 b=1 => {:#?}", model.clone().forward(vec![1.0, 1.0]));
}
