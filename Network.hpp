#include "neural_network.hpp"

class Network
{
private:
    std::vector<Layer> layers; // includes endlayer but not beginlayer
    std::vector<uint32_t> layer_sizes;

    long double cost_for_training_data_point(std::vector<long double> output, std::vector<long double> expected_output)
    {
        long double cost = 0;
        for (uint32_t i = 0; i < output.size(); i++)
        {
            cost += node_cost_function(output, expected_output);
        }
        return cost;
    }

public:
    Network(uint32_t input_node_amount, std::vector<uint32_t> l_sizes)
    {
        generator.seed((uint32_t)(std::chrono::system_clock::now().time_since_epoch().count()));

        layer_sizes = l_sizes;
        layers.push_back(Layer(input_node_amount, layer_sizes[0]));
        for (uint32_t i = 1; i < layer_sizes.size(); ++i)
        {
            layers.push_back(Layer(layer_sizes[i - 1], layer_sizes[i]));
        }
    }

    Network(std::string file)
    {
        // TODO
    }

    std::vector<long double> calculate(std::vector<long double> input)
    {
        for (uint32_t i = 0; i < layer_sizes.size(); i++)
        {
            input = layers[i].calculate_layer_result(input);
        }
        return input;
    }

    void output_network()
    {
        for (uint32_t l = 0; l < layers.size(); ++l)
        {
            layers[l].output_layer();
        }
    }

    void output_network(std::string file)
    {
        // replace old file (not sure if needed):
        std::ofstream f(file);
        f.close();

        for (uint32_t l = 0; l < layers.size(); ++l)
        {
            layers[l].output_layer(file);
        }
    }

    void save_network(std::string file)
    {
        // TODO?
    }

    // returns average cost
    long double train(std::vector<TrainingDataPoint> data)
    {
        // std::vector<long double> cost(((data.size() - 1) / parameters.mini_batch_size) + 1, 0);
        uint32_t mini_batches = (((uint32_t)(data.size()) - 1) / parameters.mini_batch_size) + 1;
        long double total_costs = 0;
        for (uint32_t c = 0; c < mini_batches; c++)
        {
            std::vector<long double> next_node_derivatives = std::vector<long double>(layer_sizes[layer_sizes.size() - 1], 0);
            std::vector<long double> output;
            for (uint32_t i = c * parameters.mini_batch_size; i < (c + 1) * parameters.mini_batch_size && i < data.size(); i++)
            {
                output = calculate(data[i].input);
                total_costs += cost_for_training_data_point(output, data[i].expected_output);
                for (uint32_t o = 0; o < output.size(); ++o)
                {
                    next_node_derivatives[o] += node_cost_derivative(output[o], data[i].expected_output[o]);
                }
            }
            if (c == mini_batches - 1)
            {
                uint32_t last_batch_size = (uint32_t)(data.size()) - (mini_batches - 1) * parameters.mini_batch_size;
                for (uint32_t o = 0; o < output.size(); ++o)
                {
                    next_node_derivatives[o] /= last_batch_size;
                }
            }
            else
            {
                for (uint32_t o = 0; o < output.size(); ++o)
                {
                    next_node_derivatives[o] /= parameters.mini_batch_size;
                }
            }
            for (int l = (int)(layers.size()) - 1; l >= 0; --l)
            {
                next_node_derivatives = layers[l].back_propagation(next_node_derivatives);
            }
        }
        parameters.initial_learning_rate *= parameters.learn_rate_decay;
        return total_costs / data.size();
    }

    long double average_cost_of_training_data(std::vector<TrainingDataPoint> data)
    {
        long double cost = 0;
        for (uint32_t i = 0; i < data.size(); i++)
        {
            cost += cost_for_training_data_point(calculate(data[i].input), data[i].expected_output);
        }
        return cost / data.size();
    }
};