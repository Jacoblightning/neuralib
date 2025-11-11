/// A struct to store data for the neural network
#[derive(Debug)]
pub struct DataValue {
	/// The input value for the neural network
	pub input: Vec<f64>,
	/// The expected output for that input value
	pub expected_output: Vec<f64>,
}

impl DataValue {
	#[cfg(feature = "idx")]
	/// Create a Vec<DataValue> from 2 idx files. One for the inputs, and one for the labels.
	///
	/// Expectations:
	///		The data is stored as vec of the MSI (Most significant index) in the idx. The rest will be flattened
	/// 	The flattened label vec will have the same length as the data one.
	pub fn from_data_label_idx(input_idx: &mut (impl std::io::Read + std::io::Seek), label_idx: &mut (impl std::io::Read + std::io::Seek)) -> crate::error::Result<()> {
		use idx_lib::*;

		// Fun chained iterator shenanigans
		let data = read_idx(input_idx)?;
		let labels: Vec<usize> = read_idx(label_idx)?
			// Just here to assert that labels is 1d
			.flatten()
			// Convert to vec
			.to_vec()
			// Convert to f64 and back 
			// (f64 is the only thing we can always 100% convert to. Now we convert them back)
			.iter()
			.map(|x| x.cast_as::<f64>().unwrap() as usize)
			.collect();

		let output_length = *labels.iter().max().unwrap();
		let out_vec = vec![0.0; output_length+1];

		let data_parsed = data
			// Iterate through all of the actual data values
			.outer_iter()
			// Flatten each of them (to prep them to be inputs) and convert to a Vec
			.map(|x| x.flatten().to_vec())
			// Convert to f64s
			.map(|x: Vec<_>| x.iter().map(|y| y.cast_as::<f64>().unwrap()).collect::<Vec<_>>())
			// Combine them with the labels
			.zip(labels.iter())
			// Convert to DataValues
			.map(|(ip, lab)| {
				let mut out =  out_vec.clone();
				out[*lab] = 1.0;
				DataValue {
					input: ip,
					expected_output: out
				}
			})
			.collect::<Vec<_>>();

		//let data_parsed = data_parsed[0].flatten();

		println!("{data_parsed:#?}");
		Ok(())
	}
}
