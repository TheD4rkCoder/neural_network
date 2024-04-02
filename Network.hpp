#include "neural_network.hpp"

class Network
{
private:
    std::vector<Layer> layers; // includes endlayer but not beginlayer
    std::vector<int> layer_sizes;

    double cost_for_training_data_point(std::vector<double> output, std::vector<double> expected_output)
    {
        double cost = 0;
        for (int i = 0; i < output.size(); i++)
        {
            cost += node_cost_function(output, expected_output);
        }
        return cost;
    }

public:
    Network(int input_node_amount, std::vector<int> layer_sizes)
    {
        generator.seed(std::chrono::system_clock::now().time_since_epoch().count());

        this->layer_sizes = layer_sizes;
        layers.push_back(Layer(input_node_amount, layer_sizes[0]));
        for (int i = 1; i < layer_sizes.size(); ++i)
        {
            layers.push_back(Layer(layer_sizes[i - 1], layer_sizes[i]));
        }
    }
    Network(std::string file)
    {
        // TODO
    }

    std::vector<double> calculate(std::vector<double> input)
    {
        for (int i = 0; i < layer_sizes.size(); i++)
        {
            input = layers[i].calculate_layer_result(input);
        }
        return input;
    }
    double average_cost(std::vector<Training_data_point> data)
    {
        double cost = 0;
        for (int i = 0; i < data.size(); i++)
        {
            cost += cost_for_training_data_point(calculate(data[i].input), data[i].expected_output);
        }
        return cost / data.size();
    }
    void output_network()
    {
        for (int l = 0; l < layers.size(); ++l)
        {
            layers[l].output_layer();
        }
    }
    void output_network(std::string file)
    {
        // replace old file:
        std::ofstream f(file);
        f.close();

        for (int l = 0; l < layers.size(); ++l)
        {
            layers[l].output_layer(file);
        }
    }

    void save_network(std::string file)
    {
        // TODO
    }
    double train(std::vector<Training_data_point> data)
    {
        double cost = 0;
        for (auto d : data)
        {
            std::vector<double> output = calculate(d.input);
            cost += cost_for_training_data_point(output, d.expected_output);

            std::vector<double> next_node_derivatives = std::vector<double>(output.size());
            for (int o = 0; o < output.size(); ++o)
            {
                next_node_derivatives[o] = node_cost_derivative(output[o], d.expected_output[o]);
            }
            for (int l = layers.size() - 1; l >= 0; --l)
            {
                next_node_derivatives = layers[l].back_propagation(next_node_derivatives);
            }
        }
        parameters.initial_learning_rate *= parameters.learn_rate_decay;

        return cost / data.size();
        /*
        // with difference quotient (very inefficient):
        double h = 0.0001;
        double original_cost = average_cost(data);
        for (Layer l : layers)
        {
            for (int out = 0; out < l.output_nodes_amount; ++out)
            {
                for (int in = 0; in < l.input_nodes_amount; ++in)
                {
                    l.weights[in][out] += h;
                    double delta_cost = average_cost(data) - original_cost;
                    l.weights[in][out] += h;
                    l.weight_cost_gradient[in][out] = delta_cost / h;
                }
            }
            for (int out = 0; out < l.output_nodes_amount; ++out)
            {
                    l.biases[out] += h;
                    double delta_cost = average_cost(data) - original_cost;
                    l.biases[out] += h;
                    l.bias_cost_gradient[out] = delta_cost / h;
            }
        }
        //*/
    }
};