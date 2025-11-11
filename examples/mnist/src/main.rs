use neuralib::{network::NeuralNetwork, activation::Activation, training::DataValue};
use std::{io::BufReader, fs::File};
use indicatif::ProgressBar;

fn main() {
    let epochs = 100000;
    
    let mut input_idx = BufReader::new(File::open("src/train-images-idx3-ubyte").unwrap());
    let mut labels_idx = BufReader::new(File::open("src/train-labels-idx1-ubyte").unwrap());
    
    let data: Vec<DataValue> = DataValue::from_data_label_idx(&mut input_idx, &mut labels_idx).unwrap();

    // Network with 784 inputs, 100 hidden, and 10 outputs. Both the hidden layer and the output have sigmoid activation
    let mut network = NeuralNetwork::new(&[100, 10], 784, vec![Activation::Sigmoid, Activation::Sigmoid]).unwrap();

    let bar = ProgressBar::new(epochs);

    println!("Learning...");
    for epoch in 0..epochs {
        bar.inc(1);
        network.learn(&data, 0.5);
        println!("Epoch: {epoch}. Loss: {}", network.loss(&data).unwrap());
    }
}
