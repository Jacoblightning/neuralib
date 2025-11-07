use std::error;
use std::fmt;

pub type Result<T> = std::result::Result<T, Box<dyn error::Error>>;


#[derive(Debug, Clone)]
pub struct InputSizeError {
	pub inputted: usize,
	pub expected: usize,
	pub chain_depth: String,
}

#[derive(Debug, Clone)]
pub struct NoLayersError {}


impl fmt::Display for InputSizeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Incorrect size passed to {}. Expected {} inputs, got {}.", self.chain_depth, self.expected, self.inputted)
    }
}

impl fmt::Display for NoLayersError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "No layer sizes given.")
    }
}

impl error::Error for InputSizeError {}
impl error::Error for NoLayersError {}
