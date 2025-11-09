use rand::prelude::*;
use rand_distr::StandardNormal;
use crate::activation::Activation;

#[derive(Debug, Default)]
pub struct LossGradient {
    pub loss_gradient_weight: Vec<f64>,
    pub loss_gradient_bias: f64,
}

#[derive(Debug)]
pub struct Neuron {
    weights: Vec<f64>,
    bias: f64,
    input_size: usize,
    activation: Activation,
    loss_gradient: LossGradient,
}

impl Neuron {
    pub fn new(input_size: usize, activation: Activation) -> Neuron {
        // Initalize weights based off of https://cs231n.github.io/neural-networks-2/#init
        let divi = (2.0 / (input_size as f64)).sqrt();
        let weights: Vec<f64> = rand::rng().sample_iter(StandardNormal).take(input_size).map(|x: f64| {x * divi}).collect();
        
        Neuron {
            weights,
            bias: 0.0,
            input_size,
            activation,
            loss_gradient: LossGradient {loss_gradient_weight: vec![0.0; input_size], loss_gradient_bias: 0.0},
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

        let activated = self.activation.call(biased);
        
        Ok(activated)
    }

    pub fn set_weight(&mut self, weight_idx: usize, new_weight: &f64) -> Result<(), ()> {
        if let Some(weight) = self.get_weight_mut(weight_idx) {
            weight.clone_from(new_weight);
            Ok(())
        } else {
            Err(())
        }
    }

    pub fn get_weight(&self, weight_idx: usize) -> Option<&f64> {
        self.weights.get(weight_idx)
    }

    pub fn get_weight_mut(&mut self, weight_idx: usize) -> Option<&mut f64> {
        self.weights.get_mut(weight_idx)
    }

    pub fn set_bias(&mut self, new_bias: &f64) {
        self.get_bias_mut().clone_from(new_bias)
    }

    pub fn get_bias(&self) -> &f64 {
        &self.bias
    }
    
    pub fn get_bias_mut(&mut self) -> &mut f64 {
        &mut self.bias
    }

    pub fn get_loss_gradient_mut(&mut self) -> &mut LossGradient {
        &mut self.loss_gradient
    }
    
    pub fn get_weight_count(&self) -> usize {
        self.input_size
    }

    pub fn apply_gradients(&mut self, learn_rate: f64) {
        self.bias -= self.loss_gradient.loss_gradient_bias * learn_rate;
        for idx in 0..self.get_weight_count() {
            self.weights[idx] -= self.loss_gradient.loss_gradient_weight[idx] * learn_rate;
        }
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
            activation: Activation::Linear,
            loss_gradient: LossGradient::default(),
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
            activation: Activation::Linear,
            loss_gradient: LossGradient::default(),
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
            activation: Activation::Linear,
            loss_gradient: LossGradient::default(),
        };
        let neuron2 = Neuron {
            weights: vec![1.0, 1.0],
            bias: 0.0,
            input_size: 2,
            activation: Activation::Linear,
            loss_gradient: LossGradient::default(),
        };

        assert!(neuron1.activate(&vec![0.0, 0.0]).is_err());
        assert!(neuron2.activate(&vec![0.0]).is_err());

        assert!(neuron1.activate(&vec![0.0]).is_ok());
        assert!(neuron2.activate(&vec![0.0, 0.0]).is_ok());
    }

    #[test]
    fn methods() {
        // Test for new method
        let mut neuron = Neuron::new(2, Activation::Sigmoid);

        for i in -100..=100 {
            // Bias stuff
            neuron.set_bias(&(i as f64));
            assert_eq!(*neuron.get_bias(), i as f64);
            let nonmut = *neuron.get_bias();
            let mutt = *neuron.get_bias_mut();
            assert_eq!(nonmut, mutt);

            // Weight stuff
            for weightidx in 0..neuron.get_weight_count() {
                neuron.set_weight(weightidx, &(i as f64)).unwrap();
                assert_eq!(*neuron.get_weight(weightidx).unwrap(), i as f64);
                let nonmut = *neuron.get_weight(weightidx).unwrap();
                let mutt = *neuron.get_weight_mut(weightidx).as_deref().unwrap();
                assert_eq!(nonmut, mutt);
            }
        }

        assert!(neuron.set_weight(neuron.get_weight_count(), &0.0).is_err());
    }
}
