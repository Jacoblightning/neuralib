//! Easy neural network library

mod error;

/// Module containing activation functions for a neural network
pub mod activation;
mod layer;
/// Module for creating, training, and running a neural network
pub mod network;
mod neuron;
/// Module containing useful structs for training and training data
pub mod training;
