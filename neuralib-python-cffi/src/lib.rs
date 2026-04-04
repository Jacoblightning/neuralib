use pyo3::prelude::*;
use pyo3::exceptions::PyException;


// Converting Box<dyn std::error::Error> into a PyErr
struct ErrorMessage {
    message: String
}

impl From<Box<dyn std::error::Error>> for ErrorMessage {
    fn from(error: Box<dyn std::error::Error>) -> Self {
        Self { message: error.to_string() }
    }
}

impl From<ErrorMessage> for PyErr {
    fn from(error: ErrorMessage) -> Self {
        // Just throw a generic exception
        PyException::new_err(error.message)
    }
}

type PyAnyResult<T> = std::result::Result<T, ErrorMessage>;

/// A Python module implemented in Rust.
#[pymodule]
mod neuralib_rs_cffi {
    use pyo3::prelude::*;

    /// Network module CFFI
    #[pymodule]
    mod network {
        use pyo3::prelude::*;

        #[pyclass]
        struct NeuralNetwork {
            inner_network: neuralib::network::NeuralNetwork
        }

        #[pymethods]
        impl NeuralNetwork {
            #[new]
            fn __new__(layer_sizes: Vec<usize>, activation_functions: Vec<super::activation::Activation>) -> crate::PyAnyResult<Self> {
                Ok(NeuralNetwork { inner_network: neuralib::network::NeuralNetwork::new(&layer_sizes, activation_functions.into_iter().map(|func| func.into()).collect())? })
            }

            fn activate(&mut self, inputs: Vec<f64>) -> crate::PyAnyResult<Vec<f64>> {
                Ok(self.inner_network.activate(&inputs)?)
            }

            fn get_layer_count(&self) -> usize {
                self.inner_network.get_layer_count()
            }

            fn loss_with_value(&mut self, value: &super::training::DataValue) -> crate::PyAnyResult<f64> {
                Ok(self.inner_network.loss_with_value(&value.inner_value)?)
            }

            fn loss(&mut self, values: Vec<super::training::DataValue>) -> crate::PyAnyResult<f64> {
                Ok(self.inner_network.loss(
                    &values.into_iter().map(|x| x.inner_value).collect::<Vec<_>>()
                )?)
            }

            fn learn(&mut self, training_data: Vec<super::training::DataValue>, learn_rate: f64) -> crate::PyAnyResult<()> {
                Ok(self.inner_network.learn(
                    &training_data.into_iter().map(|x| x.inner_value).collect::<Vec<_>>(),
                    learn_rate
                )?)
            }

            fn learn_randomly(&mut self, training_data: Vec<super::training::DataValue>, learn_rate: f64, amount: usize) -> crate::PyAnyResult<()> {
                Ok(self.inner_network.learn_randomly(
                    &training_data.into_iter().map(|x| x.inner_value).collect::<Vec<_>>(),
                    learn_rate,
                    amount
                )?)
            }

        }
    }

    #[pymodule]
    mod activation {
        use pyo3::prelude::*;

        #[pyclass]
        #[derive(Clone)]
        pub enum Activation {
            /// A linear activation function. The output is the same as the input
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
        }

        impl From<Activation> for neuralib::activation::Activation {
            fn from(a: Activation) -> Self {
                match a {
                    Activation::Linear => neuralib::activation::Activation::Linear,
                    Activation::Step => neuralib::activation::Activation::Step,
                    Activation::Sigmoid => neuralib::activation::Activation::Sigmoid,
                    Activation::HyperTan => neuralib::activation::Activation::HyperTan,
                    Activation::SiLU => neuralib::activation::Activation::SiLU,
                    Activation::ReLU => neuralib::activation::Activation::ReLU,
                    Activation::LeakyReLU => neuralib::activation::Activation::LeakyReLU,
                }
            }
        }
    }

    #[pymodule]
    mod training {
        use pyo3::prelude::*;

        #[pyclass]
        #[derive(Clone)]
        pub struct DataValue {
            pub inner_value: neuralib::training::DataValue
        }

        #[pymethods]
        impl DataValue {
            #[new]
            fn __new__(input: Vec<f64>, expected_output: Vec<f64>) -> Self {
                Self { inner_value: neuralib::training::DataValue {
                        input,
                        expected_output
                }}
            }
        }
    }
}
