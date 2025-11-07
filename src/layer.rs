use crate::neuron::Neuron;

#[derive(Clone)]
pub struct Layer {
	neurons: Vec<Neuron>,
	input_size: usize
}

impl Layer {
	pub fn new(input_size: usize, layer_size: usize) -> Layer {
		Layer {
			neurons: vec![Neuron::new(input_size); layer_size],
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
