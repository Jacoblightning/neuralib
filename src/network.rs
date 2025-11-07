use crate::layer::Layer;
use fastnum::D64;

pub struct NeuralNetwork {
	layers: Vec<Layer>,
	input_size: usize,
}

impl NeuralNetwork {
	pub fn new(layer_sizes: &[usize], input_size: usize) -> crate::error::Result<NeuralNetwork> {
		if layer_sizes.is_empty() {
			return Err(crate::error::NoLayersError {}.into());
		}
	
		// Allocate a vector for the layers
		let mut layers: Vec<Layer> = Vec::with_capacity(layer_sizes.len());


		let mut previous_size = &input_size;
		for layer_size in layer_sizes {
			layers.push(Layer::new(*previous_size, *layer_size));
			previous_size = layer_size;
		}

		Ok(NeuralNetwork {
			layers,
			input_size
		})
	}

	pub fn activate(&self, inputs: &[D64]) -> crate::error::Result<Vec<D64>> {
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
}
