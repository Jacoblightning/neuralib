use crate::neuron::Neuron;
use crate::activation::Activation;

#[derive(Debug)]
pub struct Layer {
	neurons: Vec<Neuron>,
	neuron_count: usize,
	input_size: usize,
}

impl Layer {
	pub fn new(input_size: usize, layer_size: usize, activation: Activation) -> Layer {
		Layer {
			neuron_count: layer_size,
			neurons: (0..layer_size).map(|_| Neuron::new(input_size, activation.clone())).collect(),
			input_size,
		}
	}

	pub fn activate(&mut self, inputs: &[f64]) -> crate::error::Result<Vec<f64>> {
		if inputs.len() != self.input_size {
            return Err(crate::error::InputSizeError {
                    inputted: inputs.len(),
                    expected: self.input_size,
                    chain_depth: "Layer".to_owned()
                }.into()
            );
        }

        Ok(self.neurons.iter_mut()
        	.map(|neuron| neuron.activate(inputs).expect("Length was already checked. This should not fail. (Layer)"))
        	.collect())
	}

	pub fn update_gradients_output(&mut self, expected_outputs: &[f64]) {
		for (neuron, output) in self.neurons.iter_mut().zip(expected_outputs) {
			neuron.calculate_deriv_output(output);
			neuron.update_gradients();
		}
	}

	pub fn update_gradients_hidden(&mut self, next_layer: &Layer) {
		for (neuronidx, neuron) in self.neurons.iter_mut().enumerate() {
			neuron.calculate_deriv_hidden(next_layer, neuronidx);
			neuron.update_gradients();
		}
	}

	pub fn get_neuron_count(&self) -> usize {
		self.neuron_count
	}

	pub fn get_neuron(&self, idx: usize) -> Option<&Neuron> {
		self.neurons.get(idx)
	}

	pub fn get_neuron_mut(&mut self, idx: usize) -> Option<&mut Neuron> {
		self.neurons.get_mut(idx)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn layer() {
		let mut layer = Layer::new(1, 2, crate::activation::Activation::Linear);

		assert!(layer.activate(&[]).is_err());
		layer.activate(&[1.0]).unwrap();

		assert_eq!(layer.get_neuron_count(), 2);
	}
}
