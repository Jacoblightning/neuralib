use neuralib::{network::NeuralNetwork, activation::Activation, training::DataValue};
use std::{io::BufReader, fs::File};
use indicatif::ProgressBar;

fn main() {
    let epochs = 650;
    
    let mut input_idx = BufReader::new(File::open("src/train-images-idx3-ubyte").unwrap());
    let mut labels_idx = BufReader::new(File::open("src/train-labels-idx1-ubyte").unwrap());

    let mut test_input = BufReader::new(File::open("src/t10k-images-idx3-ubyte").unwrap());
    let mut test_labels = BufReader::new(File::open("src/t10k-labels-idx1-ubyte").unwrap());
    
    let data: Vec<DataValue> = DataValue::from_data_label_idx(&mut input_idx, &mut labels_idx, None).unwrap();
    let test_data: Vec<DataValue> = DataValue::from_data_label_idx(&mut test_input, &mut test_labels, None).unwrap();

    // Network with 784 inputs, 100 hidden, and 10 outputs. Both the hidden layer and the output have sigmoid activation
    let mut network = NeuralNetwork::new(&[100, 10], 784, vec![Activation::Sigmoid, Activation::Sigmoid]).unwrap();

    let bar = ProgressBar::new(epochs);

    let epoch_size = data.len() / 10;

    println!("Learning... (Epoch size: {epoch_size})");
    for epoch in 1..=epochs {
        bar.inc(1);
        network.learn_randomly(&data, 0.5, epoch_size).unwrap();
        if epoch % 100 == 0 {
            network.save(&mut File::create(format!("save-epoch-{epoch}.mp")).unwrap()).unwrap();
            println!("Epoch: {epoch}. (Saved). Loss: {}", network.loss(&test_data).unwrap());
        } else if epoch % 20 == 0 {
            network.save(&mut File::create(format!("save-epoch-{epoch}.mp")).unwrap()).unwrap();
            println!("Epoch: {epoch}. (Saved)");
        } else {
            println!("Epoch: {epoch}.");
        }
    }

    network.save(&mut File::create("final.mp").unwrap()).unwrap();

    //println!("{network:#?}");
}
