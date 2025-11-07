use fastnum::{dec64, D64};

#[derive(Clone)]
pub struct Neuron {
    weights: Vec<D64>,
    bias: D64,
    //output: dyn Fn(D64) -> D64,
}

impl Neuron {
    pub fn new(input_size: usize) -> Neuron {
        Neuron {
            weights: vec![dec64!(0); input_size],
            bias: dec64!(0),
        }
    }
    
    pub fn activate(&self, inputs: &[D64]) -> crate::error::Result<D64> {
        if inputs.len() != self.weights.len() {
            return Err(crate::error::InputSizeError {
                    inputted: inputs.len(),
                    expected: self.weights.len(),
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
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_neuron() {
        let neuron = Neuron {
            weights: vec![dec64!(1)],
            bias: dec64!(0),
        };


        assert_eq!(neuron.activate(&vec![dec64!(0)]).unwrap(), dec64!(0));
        assert_eq!(neuron.activate(&vec![dec64!(1)]).unwrap(), dec64!(1));
        assert_eq!(neuron.activate(&vec![dec64!(123)]).unwrap(), dec64!(123));
        assert_eq!(neuron.activate(&vec![dec64!(-50)]).unwrap(), dec64!(-50));
        assert_eq!(neuron.activate(&vec![dec64!(-0)]).unwrap(), dec64!(-0));
        assert_eq!(neuron.activate(&vec![dec64!(-1)]).unwrap(), dec64!(-1));
    }

    #[test]
    fn advanced_neuron() {
        let neuron = Neuron {
            weights: vec![dec64!(2), dec64!(3)],
            bias: dec64!(-1),
        };


        assert_eq!(neuron.activate(&vec![dec64!(3), dec64!(2)]).unwrap(), dec64!(11));
        assert_eq!(neuron.activate(&vec![dec64!(8), dec64!(2)]).unwrap(), dec64!(21));
        assert_eq!(neuron.activate(&vec![dec64!(0), dec64!(0)]).unwrap(), dec64!(-1));
        assert_eq!(neuron.activate(&vec![dec64!(1), dec64!(1)]).unwrap(), dec64!(4));
        assert_eq!(neuron.activate(&vec![dec64!(-4), dec64!(-1)]).unwrap(), dec64!(-12));
    }

    #[test]
    fn size_matching() {
        let neuron1 = Neuron {
            weights: vec![dec64!(1)],
            bias: dec64!(0),
        };
        let neuron2 = Neuron {
            weights: vec![dec64!(1), dec64!(1)],
            bias: dec64!(0),
        };

        assert!(neuron1.activate(&vec![dec64!(0), dec64!(0)]).is_err());
        assert!(neuron2.activate(&vec![dec64!(0)]).is_err());

        assert!(neuron1.activate(&vec![dec64!(0)]).is_ok());
        assert!(neuron2.activate(&vec![dec64!(0), dec64!(0)]).is_ok());
    }
}
