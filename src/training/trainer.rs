use crate::training::batch::{Batch, FittedBatch};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use model::types::NetworkConfig;
use std::time::Instant;

pub trait Trainable {
    fn step(&mut self, batch: &Batch) -> FittedBatch;
}

pub struct Trainer<T: Trainable> {
    batch_size: u64,
    batch: Batch,
    strategy: T,
}

impl<T: Trainable> Trainer<T> {
    pub fn new(batch_size: u64, network_config: NetworkConfig, strategy: T) -> Trainer<T> {
        Trainer {
            batch_size,
            batch: Batch::with_population(batch_size, network_config),
            strategy,
        }
    }

    fn epoch(&mut self, steps: u64) -> FittedBatch {
        let sty =
            ProgressStyle::with_template("[{elapsed_precise}] {bar:30.cyan/blue} {pos}/{len} ")
                .unwrap()
                .progress_chars("#>-");
        let progress = ProgressBar::new(steps);
        progress.set_style(sty);

        // First step
        let mut fitted_batch = self.strategy.step(&self.batch);

        self.batch = fitted_batch.clone().into();

        progress.inc(1);

        // Remaining steps
        for _ in 1..steps {
            fitted_batch = self.strategy.step(&self.batch);

            self.batch = fitted_batch.clone().into();

            progress.inc(1);
        }

        // Finish the progress bar
        progress.finish_and_clear();

        // Return the final fitted batch
        fitted_batch
    }

    pub fn train(&mut self, epochs: u64, steps: u64) -> FittedBatch {
        // First epoch
        let now = Instant::now();
        let mut fitted_batch = self.epoch(steps);
        let mut last_fitness = fitted_batch.get_best_fitness().unwrap_or(f32::MAX);

        Self::print_epoch_stats(
            1,
            epochs,
            steps,
            now.elapsed().as_secs_f32(),
            last_fitness,
            f32::MAX,
        );

        // Remaining epochs
        for epoch in 1..epochs {
            let now = Instant::now();

            fitted_batch = self.epoch(steps);

            let elapsed = now.elapsed().as_secs_f32();

            let current_fitness = fitted_batch.get_best_fitness().unwrap_or(last_fitness);

            Self::print_epoch_stats(
                epoch + 1,
                epochs,
                steps,
                elapsed,
                current_fitness,
                last_fitness,
            );

            last_fitness = current_fitness;
        }

        fitted_batch
    }

    fn print_epoch_stats(
        epoch: u64,
        epochs: u64,
        steps: u64,
        elapsed_secs: f32,
        current_fitness: f32,
        last_fitness: f32,
    ) {
        let delta = if last_fitness > current_fitness {
            style(format!(
                "-{:.2}%",
                (1.0 - current_fitness / last_fitness) * 100.0
            ))
            .green()
        } else if last_fitness == current_fitness {
            style("±0.00%".to_string()).yellow()
        } else {
            style(format!(
                "+{:.2}%",
                (current_fitness / last_fitness - 1.0) * 100.0
            ))
            .red()
        };

        println!(
            "Epoch {}/{} \tRate {:.2}/s \tFitness {} {}",
            epoch,
            epochs,
            steps as f32 / elapsed_secs,
            current_fitness,
            delta
        );
    }
}
