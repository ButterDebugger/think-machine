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
        0.1,
        vec![2, 3, 3],
        1,
        vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ],
    );

    trainer.train(10, 100);

    let net = trainer.batch.eval_fitness(vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ]);

    // println!("{:#?}", net.iter().map(|l| l.0).collect::<Vec<_>>());

    // Test the best network
    println!();
    println!("best {:#?}", net[0].0);
    println!("0 xor 0 => {:#?}", net[0].1.clone().forward(vec![0.0, 0.0]));
    println!("0 xor 1 => {:#?}", net[0].1.clone().forward(vec![0.0, 1.0]));
    println!("1 xor 0 => {:#?}", net[0].1.clone().forward(vec![1.0, 0.0]));
    println!("1 xor 1 => {:#?}", net[0].1.clone().forward(vec![1.0, 1.0]));
}
