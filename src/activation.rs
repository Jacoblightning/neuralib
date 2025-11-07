#[derive(Clone)]
pub enum Activation {
	Linear,
	Step,
	Sigmoid,
	HyperTan,
	SiLU,
	ReLU,
	LeakyReLU,
	Swish
}

impl Default for Activation {
	fn default() -> Activation {
		Activation::Linear
	}
}

impl Activation {
	pub fn call(&self, x: f64) -> f64 {
		match self {
			Activation::Linear    => Activation::linear(x),
			Activation::Step      => Activation::step(x),
			Activation::Sigmoid   => Activation::sigmoid(x),
			Activation::HyperTan  => Activation::hypertan(x),
			Activation::SiLU      => Activation::si_lu(x),
			Activation::ReLU      => Activation::re_lu(x),
			Activation::LeakyReLU => Activation::leaky_re_lu(x),
			Activation::Swish     => Activation::swish(x),
		}
	}


	fn linear(x: f64) -> f64 {
		x
	}
	
	fn step(x: f64) -> f64 {
		if x>0.0 {1.0} else {0.0}
	}
	
	fn sigmoid(x: f64) -> f64 {
		(1.0 + (-x).exp()).recip()
	}
	
	fn hypertan(x: f64) -> f64 {
		x.tanh()
	}
	
	fn si_lu(x: f64) -> f64 {
		x / (1.0 + (-x).exp())
	}
	
	fn re_lu(x: f64) -> f64 {
		x.max(0.0)
	}

	fn leaky_re_lu(x: f64) -> f64 {
		x.max(x * 0.15)
	}

	fn swish(x: f64) -> f64 {
		let beta = 1.0;
		x * Activation::sigmoid(beta * x)
	}
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear() {
        let act = Activation::Linear;

        for i in 0..=100 {
        	assert_eq!(act.call(i as f64), i as f64);
        }
    }

    #[test]
    fn test_sigmoid() {
        let act = Activation::Sigmoid;
        assert_eq!(act.call(0.0), 0.5);
    }
}
