use crate::layer::Layer;
use crate::activation::Activation;
use crate::training::DataValue;

#[derive(Debug)]
pub struct NeuralNetwork {
	layers: Vec<Layer>,
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
			layers,
			input_size
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

	pub fn loss_with_value(&self, value: DataValue) -> crate::error::Result<f64> {
		let output = self.activate(&value.input)?;

		let mut loss = 0.0;

		for (actual, expected) in output.iter().zip(value.expected_output) {
			loss += actual - expected
		}

		Ok(loss)
	}

	pub fn loss(&self, values: Vec<DataValue>) -> crate::error::Result<f64> {
		let mut total_loss = 0.0;

		let value_length = values.len();

		for value in values {
			total_loss += self.loss_with_value(value)?;
		}

		Ok(total_loss / (value_length as f64))
	}
}
