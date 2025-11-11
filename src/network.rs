use crate::layer::Layer;
use crate::neuron::Neuron;
use crate::activation::Activation;
use crate::training::DataValue;

/// A neural network
#[derive(Debug)]
pub struct NeuralNetwork {
	layers: Vec<Layer>,
	layer_count: usize,
	input_size: usize,
}

impl NeuralNetwork {
	/// Create a new neural network
	///
	/// Arguments:
	///
	/// * `layer_sizes` - A slice of usizes containing the size of each layer in the neural network
	/// * `input_size`  - How many inputs the first layer should accept
	/// * `activation_functions` - A Vec of which activation function should be in each layer 
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

	/// Run the neural network with specific inputs
	///
	/// Arguments:
	///
	/// * `inputs` - A slice of f64s to be used as input to the network
	pub fn activate(&mut self, inputs: &[f64]) -> crate::error::Result<Vec<f64>> {
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

        for layer in &mut self.layers {
        	// All the sizes *should* be correct
        	next_in = layer.activate(&next_in).expect("Length was already checked. This should not fail. (Network)")
        }

        Ok(next_in)
	}

	/// Get the number of layers in this neural network
	pub fn get_layer_count(&self) -> usize {
		self.layer_count
	}

	fn get_layer(&self, idx: usize) -> Option<&Layer> {
		self.layers.get(idx)
	}

	fn get_layer_mut(&mut self, idx: usize) -> Option<&mut Layer> {
		self.layers.get_mut(idx)
	}

	/// Calculate the loss of the network with a DataValue
	///
	/// Arguments:
	///
	/// * `value` - A reference to a DataValue
	pub fn loss_with_value(&mut self, value: &DataValue) -> crate::error::Result<f64> {
		let output = self.activate(&value.input)?;

		let mut loss = 0.0;

		for (actual, expected) in output.iter().zip(value.expected_output.iter()) {
			loss += Neuron::loss(actual, expected);
		}

		Ok(loss)
	}

	/// Calculate the average loss for a slice of DataValues.
	/// This method should be preferred over `loss_with_value`
	///
	/// Arguments:
	///
	/// * `values` - A slice of DataValues to test
	pub fn loss(&mut self, values: &[DataValue]) -> crate::error::Result<f64> {
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

	/// Train the network on some data
	///
	/// Arguments:
	///
	/// * `training_data` - The data to train the network on in a slice of DataValues
	/// * `learn_rate` - How fast the network should try to learn
	pub fn learn(&mut self, training_data: &[DataValue], learn_rate: f64) {
		for value in training_data {
			self.update_all_gradients(value);
		}

		self.apply_gradients(learn_rate / (training_data.len() as f64));
	}


	pub fn learn_randomly(&mut self, training_data: &[DataValue], learn_rate: f64, amount: usize) {
		use rand::seq::SliceRandom;
		let mut rand_split = training_data.to_vec();

		// Shuffle the data
		rand_split.shuffle(&mut rand::rng());
		// Get the split
		self.learn(&rand_split[..amount], learn_rate)
	}

	fn update_all_gradients(&mut self, value: &DataValue) {
		// Prep the network
		self.activate(&value.input).expect("Length was already checked. This should not fail. (Network)");

		let output_layer = self.get_layer_mut(self.get_layer_count() - 1).expect("Length was already checked. This should not fail. (Network)");
		output_layer.update_gradients_output(&value.expected_output);
		
		for layeridx in (0..self.get_layer_count()).rev().skip(1) {
			// Fun borrow checker shenanigans
			let (up_to_current, past_current) = self.layers.split_at_mut_checked(layeridx+1).expect("Length was already checked. This should not fail. (Network)");
			let current_layer = up_to_current.get_mut(layeridx).expect("Length was already checked. This should not fail. (Network)");
			let next_layer = past_current.first().expect("Length was already checked. This should not fail. (Network)");
			current_layer.update_gradients_hidden(next_layer);
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	
	#[test]
	fn methods() {
		let mut network = NeuralNetwork::new(&[2, 2], 2, vec![Activation::Sigmoid, Activation::Step]).unwrap();

		network.activate(&[0.0, 0.0]).unwrap();

		assert_eq!(network.get_layer_count(), 2);
	}

	#[test]
	fn errors() {
		assert!(NeuralNetwork::new(&[], 0, vec![]).is_err());
		assert!(NeuralNetwork::new(&[1], 0, vec![]).is_err());

		let mut network = NeuralNetwork::new(&[1], 1, vec![Activation::Linear]).unwrap();
		assert!(network.activate(&[]).is_err());
	}
}
