
/// A struct to store data for the neural network
#[derive(Debug)]
pub struct DataValue {
	/// The input value for the neural network
	pub input: Vec<f64>,
	/// The expected output for that input value
	pub expected_output: Vec<f64>,
}
