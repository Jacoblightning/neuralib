//! Activation functions for neuralib
//!
//! This module provides many different activation functions for a neural network.

/// The activation functions this library supports
#[derive(Clone, Debug, Default)]
pub enum Activation {
	/// A linear activation function. The output is the same as the input
	#[default]
	Linear,
	/// The step activation function. The output is 0 if x<0 otherwise, it's 1
	Step,
	/// The sigmoid activation function: <https://en.wikipedia.org/wiki/Sigmoid_function>
	Sigmoid,
	/// The Hyperbolic Tangent activation function.
	HyperTan,
	/// The SiLU (Swish) activation function: <https://en.wikipedia.org/wiki/Rectified_linear_unit#SiLU>
	SiLU,
	/// The ReLU activation function: <https://en.wikipedia.org/wiki/Rectified_linear_unit>
	ReLU,
	/// The Leaky ReLU activation function: <https://en.wikipedia.org/wiki/Rectified_linear_unit#Piecewise-linear_variants>
	LeakyReLU,
	/// The Swish activation function: <https://en.wikipedia.org/wiki/Swish_function>
	#[deprecated(since="0.0.2", note="Please use SiLU instead")]
	Swish
}


impl Activation {
	/// Call the selected activation function
	pub fn call(&self, x: f64) -> f64 {
		match self {
			Activation::Linear    => Activation::linear(x),
			Activation::Step      => Activation::step(x),
			Activation::Sigmoid   => Activation::sigmoid(x),
			Activation::HyperTan  => Activation::hypertan(x),
			Activation::SiLU      => Activation::si_lu(x),
			Activation::ReLU      => Activation::re_lu(x),
			Activation::LeakyReLU => Activation::leaky_re_lu(x),
			#[allow(deprecated)]
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
		let beta = 1.0;
		x * Activation::sigmoid(beta * x)
	}
	
	fn re_lu(x: f64) -> f64 {
		x.max(0.0)
	}

	fn leaky_re_lu(x: f64) -> f64 {
		x.max(x * 0.15)
	}

	fn swish(x: f64) -> f64 {
		Activation::si_lu(x)
	}
}


#[cfg(test)]
mod tests {
    use super::*;

    fn floating_equal(a: f64, b: f64) -> bool {
    	let tolerance = 0.0001;
    	(a - b).abs() < tolerance
    }

    #[test]
    fn linear() {
        let act = Activation::Linear;

        for i in -100..=100 {
        	assert_eq!(act.call(i as f64), i as f64);
        }
    }

    #[test]
    fn step() {
    	// This should be pretty simple to test
    	let act = Activation::Step;

    	for i in -100..0 {
    		assert_eq!(act.call(i as f64), 0.0);
    	}

    	for i in 1..=100 {
    		assert_eq!(act.call(i as f64), 1.0);
    	}
    }

    #[test]
    fn sigmoid() {
        let act = Activation::Sigmoid;
        assert_eq!(act.call(0.0), 0.5);
        // Assert that the S shape is there
        assert!(act.call(9999.0) > 0.999);
        assert!(act.call(-9999.0) < 0.001);
    }

    #[test]
    fn hypertan() {
    	let act = Activation::HyperTan;
    	assert_eq!(act.call(0.0), 0.0);

    	assert!(act.call(9999.0) > 0.999);
    	assert!(act.call(-9999.0) < -0.999);
    }

    
    #[test]
    fn si_lu() {
    	let act = Activation::SiLU;
    	// This is the low point in the dip.
    	let si_lu_point = (-1.278465, -0.278465);


    	// Test the begining of ReLU
    	assert!(floating_equal(act.call(-15.0), 0.0));
    	// Test the end of ReLU
    	assert!(floating_equal(act.call(100.0), 100.0));
    	// Test the swish/SiLU point
    	assert!(floating_equal(act.call(si_lu_point.0), si_lu_point.1));
    }

    #[test]
    fn re_lu() {
    	let act = Activation::ReLU;

    	for i in -100..=0 {
    		assert_eq!(act.call(i as f64), 0.0);
    	}
    	for i in 0..=100 {
    		assert_eq!(act.call(i as f64), i as f64);
    	}
    }

    #[test]
    fn leaky_re_lu() {
    	let act = Activation::LeakyReLU;

    	for i in -100..=0 {
    		assert_eq!(act.call(i as f64), (i as f64) * 0.15);
    	}
    	for i in 0..=100 {
    		assert_eq!(act.call(i as f64), i as f64);
    	}
    }

    // Swish just calls SiLU and so doesn't need it's own test
}
