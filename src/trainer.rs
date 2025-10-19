use crate::{batch::Batch, types::Dataset};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct Trainer {
    batch_size: u64,
    learning_rate: f32,
    hidden_layer_sizes: Vec<u64>,
    output_size: u64,
    training_data: Dataset,
    last_fitness: f32,
    pub batch: Batch,
}

impl Trainer {
    pub fn new(
        batch_size: u64,
        learning_rate: f32,
        hidden_layer_sizes: Vec<u64>,
        output_size: u64,
        training_data: Dataset,
    ) -> Trainer {
        Trainer {
            batch_size,
            learning_rate,
            hidden_layer_sizes,
            output_size,
            training_data,
            last_fitness: 0.0,
            batch: Batch::new(batch_size),
        }
    }

    fn step(&mut self) {
        // Populate the batch and evaluate fitness
        self.batch
            .populate(self.hidden_layer_sizes.clone(), self.output_size);

        let sorted_batch = self.batch.clone().eval_fitness(self.training_data.clone());

        // Update the last fitness
        self.last_fitness = sorted_batch[0].0;

        // Create a new batch with the top half of the batch and their mutations
        let mut new_batch = Batch::new(self.batch_size);

        for fitted_network in sorted_batch.iter().take(self.batch_size as usize / 2usize) {
            new_batch.add_network(fitted_network.1.clone());
            new_batch.add_network(fitted_network.1.clone().mutate(self.learning_rate));
        }

        self.batch = new_batch;
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
                self.last_fitness,
            );
        }
    }
}
