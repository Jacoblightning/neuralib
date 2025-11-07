use crate::neuron::Neuron;
use crate::activation::Activation;

pub struct Layer {
	neurons: Vec<Neuron>,
	input_size: usize
}

impl Layer {
	pub fn new(input_size: usize, layer_size: usize, activation: Activation) -> Layer {
		Layer {
			neurons: (0..layer_size).map(|_| Neuron::new(input_size, activation.clone())).collect(),
			input_size,
		}
	}

	pub fn activate(&self, inputs: &[f64]) -> crate::error::Result<Vec<f64>> {
		if inputs.len() != self.input_size {
            return Err(crate::error::InputSizeError {
                    inputted: inputs.len(),
                    expected: self.input_size,
                    chain_depth: "Layer".to_owned()
                }.into()
            );
        }

        Ok(self.neurons.iter()
        	.map(|neuron| neuron.activate(inputs).expect("Length was already checked. This should not fail. (Layer)"))
        	.collect())
	}
}
