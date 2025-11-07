use fastnum::{dec64, D64};

pub struct Neuron {
    weights: Vec<D64>,
    bias: D64,
    //output: dyn Fn(D64) -> D64,
}

impl Neuron {
    pub fn new(weights: &[D64], bias: D64) -> Neuron {
        Neuron {
            weights: weights.to_vec(),
            bias,
        }
    }
    
    pub fn activate(&self, inputs: &[D64]) -> crate::error::Result<D64> {
        if inputs.len() != self.get_weight_count() {
            return Err(crate::error::InputSizeError {
                    inputted: inputs.len(),
                    expected: self.get_weight_count(),
                    chain_depth: "Neuron".to_owned()
                }.into()
            );
        }
        
        let weighted: D64 = inputs.iter()
                        // Combine weights and inputs
                        .zip(self.weights.iter())
                        // Multiply them together
                        .map(|zipped| (*zipped.0) * (*zipped.1))
                        // Sum them up
                        .sum();
        let biased = weighted + self.bias;

        // TODO: use activation/output function
        
        Ok(biased)
    }




    pub fn get_weight(&self, i: usize) -> Option<&D64> {
        self.weights.get(i)
    }

    pub fn get_weight_count(&self) -> usize {
        self.weights.len()
    }
    
    pub fn set_weight(&mut self, i: usize, weight: D64) {
        self.weights[i] = weight;
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_neuron() {
        let neuron = Neuron::new(
            &[dec64!(1)],
            dec64!(0),
        );


        assert_eq!(neuron.activate(&vec![dec64!(0)]).unwrap(), dec64!(0));
        assert_eq!(neuron.activate(&vec![dec64!(1)]).unwrap(), dec64!(1));
        assert_eq!(neuron.activate(&vec![dec64!(123)]).unwrap(), dec64!(123));
        assert_eq!(neuron.activate(&vec![dec64!(-50)]).unwrap(), dec64!(-50));
        assert_eq!(neuron.activate(&vec![dec64!(-0)]).unwrap(), dec64!(-0));
        assert_eq!(neuron.activate(&vec![dec64!(-1)]).unwrap(), dec64!(-1));
    }

    #[test]
    fn advanced_neuron() {
        let neuron = Neuron::new(
            &[dec64!(2), dec64!(3)],
            dec64!(-1),
        );


        assert_eq!(neuron.activate(&vec![dec64!(3), dec64!(2)]).unwrap(), dec64!(11));
        assert_eq!(neuron.activate(&vec![dec64!(8), dec64!(2)]).unwrap(), dec64!(21));
        assert_eq!(neuron.activate(&vec![dec64!(0), dec64!(0)]).unwrap(), dec64!(-1));
        assert_eq!(neuron.activate(&vec![dec64!(1), dec64!(1)]).unwrap(), dec64!(4));
        assert_eq!(neuron.activate(&vec![dec64!(-4), dec64!(-1)]).unwrap(), dec64!(-12));
    }

    #[test]
    fn size_matching() {
        let neuron1 = Neuron::new(
            &[dec64!(1)],
            dec64!(0),
        );
        let neuron2 = Neuron::new(
            &[dec64!(1), dec64!(1)],
            dec64!(0),
        );

        assert!(neuron1.activate(&vec![dec64!(0), dec64!(0)]).is_err());
        assert!(neuron2.activate(&vec![dec64!(0)]).is_err());

        assert!(neuron1.activate(&vec![dec64!(0)]).is_ok());
        assert!(neuron2.activate(&vec![dec64!(0), dec64!(0)]).is_ok());
    }
}
