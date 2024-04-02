#include "neural_network.hpp"

// saves an input for the first layer and expected output of the last layer
// needed for training
class Training_data_point
{
public:
    std::vector<double> input;
    std::vector<double> expected_output;
    Training_data_point();

    Training_data_point(std::vector<double> input, std::vector<double> expected_output)
    {
        this->input = input;
        this->expected_output = expected_output;
    }
};