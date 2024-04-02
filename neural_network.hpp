#ifndef NEURAL_NETWORK

#define NEURAL_NETWORK
#include <vector>
#include <math.h>
#include <string>
#include <random>
#include <chrono>
#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>

#include "Activation_functions.hpp"

struct
{
    double initial_learning_rate = 0.1;

    double learn_rate_decay = 0.99;
    int mini_batch_size = 32;
    double momentum = 0.9;
    double regularization = 0.1;
    leaky_relu_activation activation;
    // Cost
} parameters;

std::default_random_engine generator;
// Create a random engine object
// Seed the engine (optional, but recommended for better randomness)
double random_double(double min, double max)
{

    // Create a distribution object for generating random doubles within the specified range
    std::uniform_real_distribution<double> distribution(min, max);

    // Generate a random double value
    return distribution(generator);
}

#include "Layer.hpp"
#include "Training_data_point.hpp"
#include "Network.hpp"

#endif