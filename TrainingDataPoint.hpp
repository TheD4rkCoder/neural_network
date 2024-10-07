#include "neural_network.hpp"

// saves an input for the first layer and expected output of the last layer
// needed for training
class TrainingDataPoint
{
public:
    std::vector<long double> input;
    std::vector<long double> expected_output;
    TrainingDataPoint();

    TrainingDataPoint(std::vector<long double> input, std::vector<long double> expected_output)
    {
        this->input = input;
        this->expected_output = expected_output;
    }
};