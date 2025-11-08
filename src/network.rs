use crate::layer::Layer;
use crate::activation::Activation;
use crate::training::DataValue;

#[derive(Debug)]
pub struct NeuralNetwork {
	layers: Vec<Layer>,
	layer_count: usize,
	input_size: usize,
}

impl NeuralNetwork {
	pub fn new(layer_sizes: &[usize], input_size: usize, activation_functions: Vec<Activation>) -> crate::error::Result<NeuralNetwork> {
		if layer_sizes.is_empty() {
			return Err(crate::error::NoLayersError {}.into());
		}

		if layer_sizes.len() != activation_functions.len() {
			// TODO: Make custom error for this
			return Err(crate::error::NoLayersError {}.into())
		}
	
		// Allocate a vector for the layers
		let mut layers: Vec<Layer> = Vec::with_capacity(layer_sizes.len());


		let mut previous_size = &input_size;
		for (layer_size, activator) in layer_sizes.iter().zip(activation_functions) {
			layers.push(Layer::new(*previous_size, *layer_size, activator));
			previous_size = layer_size;
		}

		Ok(NeuralNetwork {
			layer_count: layers.len(),
			layers,
			input_size,
		})
	}

	pub fn activate(&self, inputs: &[f64]) -> crate::error::Result<Vec<f64>> {
		if inputs.len() != self.input_size {
            return Err(crate::error::InputSizeError {
                    inputted: inputs.len(),
                    expected: self.input_size,
                    chain_depth: "NeuralNetwork".to_owned()
                }.into()
            );
        }

        // We have to feed each layer's output into the next layer's input

        let mut next_in = inputs.to_vec();

        for layer in &self.layers {
        	// All the sizes *should* be correct
        	next_in = layer.activate(&next_in).expect("Length was already checked. This should not fail. (Network)")
        }

        Ok(next_in)
	}

	pub fn get_layer_count(&self) -> usize {
		self.layer_count
	}

	pub fn get_layer(&self, idx: usize) -> Option<&Layer> {
		self.layers.get(idx)
	}

	pub fn get_layer_mut(&mut self, idx: usize) -> Option<&mut Layer> {
		self.layers.get_mut(idx)
	}

	pub fn loss_with_value(&self, value: &DataValue) -> crate::error::Result<f64> {
		let output = self.activate(&value.input)?;

		let mut loss = 0.0;

		for (actual, expected) in output.iter().zip(value.expected_output.iter()) {
			loss += (actual - expected).powi(2);
		}

		Ok(loss)
	}

	pub fn loss(&self, values: &[DataValue]) -> crate::error::Result<f64> {
		let mut total_loss = 0.0;

		let value_length = values.len();

		for value in values {
			total_loss += self.loss_with_value(value)?;
		}

		Ok(total_loss / (value_length as f64))
	}

	fn apply_gradients(&mut self, learn_rate: f64) {
		for layeridx in 0..self.get_layer_count() {
			let layer = self.get_layer_mut(layeridx).unwrap();
			for neuronidx in 0..layer.get_neuron_count() {
				let neuron = layer.get_neuron_mut(neuronidx).unwrap();
				neuron.apply_gradients(learn_rate);
			}
		}
	}
	
	pub fn learn(&mut self, training_data: &[DataValue], learn_rate: f64) -> crate::error::Result<()> {
		let h: f64 = 0.0001;
		let starting_loss = self.loss(training_data)?;


		// This is made WAY more complicated by the borrow checker
	    for layeridx in 0..self.get_layer_count() {
			let neuron_count = self.get_layer(layeridx).unwrap().get_neuron_count();
			for neuronidx in 0..neuron_count {
				// Modify the weights for the neuron
				let weight_count = self.get_layer(layeridx).unwrap().get_neuron(neuronidx).unwrap().get_weight_count();
				for weightidx in 0..weight_count {
					// Set the weight in a seperate scope
					{
						let weight = self.get_layer_mut(layeridx).unwrap().get_neuron_mut(neuronidx).unwrap().get_weight_mut(weightidx).unwrap();
						*weight += h;
					}
					// Calculate the delta loss after dropping the mutable reference
					let delta_loss = self.loss(training_data)? - starting_loss;
					// Revert the weight (in a seperate scope) and save the data
					{
						let neuron = self.get_layer_mut(layeridx).unwrap().get_neuron_mut(neuronidx).unwrap();
						let weight = neuron.get_weight_mut(weightidx).unwrap();
						*weight -= h;
						let loss_grad = neuron.get_loss_gradient_mut();
						loss_grad.loss_gradient_weight[weightidx] = delta_loss / h;
					}
				}
				// Now modify the bias for the neuron
				{
					let bias = self.get_layer_mut(layeridx).unwrap().get_neuron_mut(neuronidx).unwrap().get_bias_mut();
					*bias += h;
				}
				// Calculate the delta loss after dropping the mutable reference
				let delta_loss = self.loss(training_data)? - starting_loss;
				// Revert the bias (in a seperate scope) and save the data
				{
					let neuron = self.get_layer_mut(layeridx).unwrap().get_neuron_mut(neuronidx).unwrap();
					let bias = neuron.get_bias_mut();
					*bias -= h;
					let loss_grad = neuron.get_loss_gradient_mut();
					loss_grad.loss_gradient_bias = delta_loss / h;
				}
	        }
	    }

	    // Now apply all of the gradients...

	    self.apply_gradients(learn_rate);

	    Ok(())
	}
}
