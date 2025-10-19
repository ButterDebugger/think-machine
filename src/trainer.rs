use crate::{batch::Batch, network::Network, types::{Dataset, NetworkConfig}};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct Trainer {
    batch_size: u64,
    learning_rate: f32,
    training_data: Dataset,
    current_fitness: f32,
    pub batch: Batch,
}

impl Trainer {
    pub fn new(
        batch_size: u64,
        learning_rate: f32,
        network_config: NetworkConfig,
        training_data: Dataset,
    ) -> Trainer {
        Trainer {
            batch_size,
            learning_rate,
            training_data,
            current_fitness: 0.0,
            batch: Batch::new_with_population(batch_size, network_config),
        }
    }

    fn step(&mut self) {
        // Evaluate fitness of the batch
        let fitted_batch = self.batch.eval_fitness(self.training_data.clone());

        // Update the last fitness
        self.current_fitness = fitted_batch[0].0;

        // Create a new batch with the top half of the batch and their mutations
        let top_count = (self.batch_size / 2) as usize;
        let mutate_count = self.batch_size as usize - top_count;

        let mut new_networks: Vec<Network> = fitted_batch
            .iter()
            .take(top_count)
            .map(|(_, network)| network.clone())
            .collect();

        new_networks.extend(
            fitted_batch
                .iter()
                .take(mutate_count)
                .map(|(_, network)| network.mutate(self.learning_rate)),
        );

        // Store the new batch of networks
        self.batch = Batch::new_with_networks(new_networks);
    }

    fn epoch(&mut self, steps: u64) {
        let sty =
            ProgressStyle::with_template("[{elapsed_precise}] {bar:30.cyan/blue} {pos}/{len} ")
                .unwrap()
                .progress_chars("#>-");
        let progress = ProgressBar::new(steps);
        progress.set_style(sty);

        for _ in 0..steps {
            // Step the batch
            self.step();

            // Update the progress bar
            progress.inc(1);
        }

        progress.finish_and_clear();
    }

    pub fn train(&mut self, epochs: u64, steps: u64) {
        for i in 0..epochs {
            let now = Instant::now();
            self.epoch(steps);
            let elapsed = now.elapsed();

            println!(
                "Epoch {}/{} \tRate {}/s \tFitness {}",
                i + 1,
                epochs,
                steps as f32 / elapsed.as_secs_f32(),
                self.current_fitness,
            );
        }
    }
}
