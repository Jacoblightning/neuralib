use rand::prelude::*;
use rand_distr::StandardNormal;

#[derive(Clone)]
pub struct Neuron {
    weights: Vec<f64>,
    bias: f64,
    input_size: usize,
    //output: dyn Fn(D64) -> D64,
}

impl Neuron {
    pub fn new(input_size: usize) -> Neuron {
        // Initalize weights based off of https://cs231n.github.io/neural-networks-2/#init
        let divi = (2.0 / (input_size as f64)).sqrt();
        let weights: Vec<f64> = rand::rng().sample_iter(StandardNormal).take(input_size).map(|x: f64| {x * divi}).collect();
        
        Neuron {
            weights,
            bias: 0.0,
            input_size,
        }
    }
    
    pub fn activate(&self, inputs: &[f64]) -> crate::error::Result<f64> {
        if inputs.len() != self.input_size {
            return Err(crate::error::InputSizeError {
                    inputted: inputs.len(),
                    expected: self.input_size,
                    chain_depth: "Neuron".to_owned()
                }.into()
            );
        }
        
        let weighted: f64 = inputs.iter()
                        // Combine weights and inputs
                        .zip(self.weights.iter())
                        // Multiply them together
                        .map(|zipped| (*zipped.0) * (*zipped.1))
                        // Sum them up
                        .sum();
        // Add the bias
        let biased = weighted + self.bias;

        // TODO: use activation/output function
        
        Ok(biased)
    }

    pub fn get_weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn set_weight(&mut self, weight_idx: usize, new_weight: f64) {
        self.weights[weight_idx] = new_weight
    }
    
    pub fn get_weight(&self, weight_idx: usize) -> Option<&f64> {
        self.weights.get(weight_idx)
    }
    
    pub fn get_input_length(&self) -> usize {
        self.input_size
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_neuron() {
        let neuron = Neuron {
            weights: vec![1.0],
            bias: 0.0,
            input_size: 1,
        };


        assert_eq!(neuron.activate(&vec![0.0]).unwrap(), 0.0);
        assert_eq!(neuron.activate(&vec![1.0]).unwrap(), 1.0);
        assert_eq!(neuron.activate(&vec![123.0]).unwrap(), 123.0);
        assert_eq!(neuron.activate(&vec![-50.0]).unwrap(), -50.0);
        assert_eq!(neuron.activate(&vec![-0.0]).unwrap(), -0.0);
        assert_eq!(neuron.activate(&vec![-1.0]).unwrap(), -1.0);
    }

    #[test]
    fn advanced_neuron() {
        let neuron = Neuron {
            weights: vec![2.0, 3.0],
            bias: -1.0,
            input_size: 2,
        };


        assert_eq!(neuron.activate(&vec![3.0, 2.0]).unwrap(), 11.0);
        assert_eq!(neuron.activate(&vec![8.0, 2.0]).unwrap(), 21.0);
        assert_eq!(neuron.activate(&vec![0.0, 0.0]).unwrap(), -1.0);
        assert_eq!(neuron.activate(&vec![1.0, 1.0]).unwrap(), 4.0);
        assert_eq!(neuron.activate(&vec![-4.0, -1.0]).unwrap(), -12.0);
    }

    #[test]
    fn size_matching() {
        let neuron1 = Neuron {
            weights: vec![1.0],
            bias: 0.0,
            input_size: 1,
        };
        let neuron2 = Neuron {
            weights: vec![1.0, 1.0],
            bias: 0.0,
            input_size: 2,
        };

        assert!(neuron1.activate(&vec![0.0, 0.0]).is_err());
        assert!(neuron2.activate(&vec![0.0]).is_err());

        assert!(neuron1.activate(&vec![0.0]).is_ok());
        assert!(neuron2.activate(&vec![0.0, 0.0]).is_ok());
    }
}
