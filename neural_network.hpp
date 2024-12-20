#ifndef NEURAL_NETWORK

#define NEURAL_NETWORK
// constants:
#include <cstdint> 

// math:
#include <cmath>
#include <random>
#include <chrono>

// data structures:
#include <string>
#include <vector>
#include <queue>

// output:
#include <iostream>
#include <fstream>

// threads:
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <atomic>

#include "Activation_functions.hpp"

struct
{
    double initial_learning_rate = 0.05;

    double learn_rate_decay = 0.999995; // for each training data point
    uint32_t mini_batch_size = 1;
    double momentum_factor = 0.8;
    double regularization = 0.1; // TBA
    leaky_relu_activation activation;
    // Cost
} parameters;

std::default_random_engine generator;
// Create a random engine object
// Seed the engine (optional, but recommended for better randomness)
long double random_ldouble(long double min, long double max)
{
    // Create a distribution object for generating random doubles within the specified range
    std::uniform_real_distribution<long double> distribution(min, max);

    // Generate a random double value
    return distribution(generator);
}

#include "Layer.hpp"
#include "TrainingDataPoint.hpp"
#include "Network.hpp"
#include "ThreadPool.hpp"

#endif