#include "neural_network.hpp"
#include <math.h> //temp
class Layer
{
private:
    // Used for adding momentum to gradient descent (not used yet D: )
    std::vector<std::vector<long double>> weight_velocities;
    std::vector<long double> bias_velocities;

    std::vector<long double> last_weighted_inputs;
    std::vector<long double> last_inputs;

    void initialize_with_random_weigths()
    {
        for (uint32_t out = 0; out < output_nodes_amount; ++out)
        {
            for (uint32_t in = 0; in < input_nodes_amount; ++in)
            {
                weights[in][out] = random_ldouble(-0.5, 0.5);
            }
            biases[out] = random_ldouble(-0.5, 0.5);
        }
    }

    void update_gradients(std::vector<long double> &node_derivatives)
    {
        for (uint32_t out = 0; out < output_nodes_amount; ++out)
        {
            for (uint32_t in = 0; in < input_nodes_amount; ++in)
            {
                weight_cost_gradient[in][out] += last_inputs[in] * node_derivatives[out];
                // if (weight_cost_gradient[in][out] > 5)
                // {
                //     weight_cost_gradient[in][out] = 5;
                // }
                // else if (weight_cost_gradient[in][out] < -5)
                // {
                //     weight_cost_gradient[in][out] = -5;
                // }
            }
            bias_cost_gradient[out] += node_derivatives[out];
            // if (bias_cost_gradient[out] > 5)
            // {
            //     bias_cost_gradient[out] = 5;
            // }
            // else if (bias_cost_gradient[out] < -5)
            // {
            //     bias_cost_gradient[out] = -5;
            // }
        }
    }

public:
    uint32_t output_nodes_amount;
    uint32_t input_nodes_amount;

    std::vector<std::vector<long double>> weights; // [from][to]
    std::vector<long double> biases;               // added to sum of weights

    std::vector<std::vector<long double>> weight_cost_gradient;
    std::vector<long double> bias_cost_gradient;

    Layer(uint32_t in_nodes_amount, uint32_t out_nodes_amount)
    {
        input_nodes_amount = in_nodes_amount;
        output_nodes_amount = out_nodes_amount;
        weights = std::vector<std::vector<long double>>(input_nodes_amount, std::vector<long double>(output_nodes_amount));
        biases = std::vector<long double>(output_nodes_amount);
        weight_cost_gradient = std::vector<std::vector<long double>>(input_nodes_amount, std::vector<long double>(output_nodes_amount, 0));
        bias_cost_gradient = std::vector<long double>(output_nodes_amount, 0);
        bias_velocities = std::vector<long double>(output_nodes_amount, 0);
        weight_velocities = std::vector<std::vector<long double>>(input_nodes_amount, std::vector<long double>(output_nodes_amount, 0));
        last_weighted_inputs = std::vector<long double>(output_nodes_amount, 0);

        initialize_with_random_weigths();
    }

    Layer(std::ifstream file)
    {
        // TODO
    }

    std::vector<long double> calculate_layer_result(std::vector<long double> &input)
    {
        std::vector<long double> output = std::vector<long double>(output_nodes_amount);
        for (uint32_t out = 0; out < output_nodes_amount; ++out)
        {
            last_weighted_inputs[out] = biases[out];
            for (uint32_t in = 0; in < input_nodes_amount; ++in)
            {
                if (isinf(input[in]))
                {
                    std::cout << "what tf " << weights[in][out] << '\n';
                    output_layer();
                    exit(-1);
                }

                last_weighted_inputs[out] += weights[in][out] * input[in]; // becomes nan here?
            }
            output[out] = parameters.activation.activate(last_weighted_inputs[out]);
        }
        last_inputs = input;
        return output;
    }

    void apply_gradient_descent()
    {
        for (uint32_t out = 0; out < output_nodes_amount; out++)
        {
            long double bias_delta = -bias_cost_gradient[out] * parameters.initial_learning_rate + bias_velocities[out] * parameters.momentum_factor;
            biases[out] += bias_delta;
            bias_velocities[out] = bias_delta;
            bias_cost_gradient[out] = 0;
            for (uint32_t in = 0; in < input_nodes_amount; in++)
            {
                long double weight_delta = -weight_cost_gradient[in][out] * parameters.initial_learning_rate + weight_velocities[in][out] * parameters.momentum_factor;
                weights[in][out] += weight_delta;
                weight_velocities[in][out] = weight_delta;
                weight_cost_gradient[in][out] = 0;
            }
        }
    }

    std::vector<long double> back_propagation(std::vector<long double> out_node_cost_derivatives)
    {
        std::vector<long double> in_node_derivatives = std::vector<long double>(input_nodes_amount, 0);

        for (uint32_t out = 0; out < output_nodes_amount; ++out)
        {
            out_node_cost_derivatives[out] *= parameters.activation.derivate(last_weighted_inputs[out]);

            for (uint32_t in = 0; in < input_nodes_amount; ++in)
            {
                in_node_derivatives[in] += out_node_cost_derivatives[out] * weights[in][out];
            }
        }
        update_gradients(out_node_cost_derivatives);
        return in_node_derivatives;
    }

    void output_layer()
    {
        std::cout << '\n'
                  << '\n';
        std::cout << "input_nodes_amount: " << input_nodes_amount << '\n';
        std::cout << "output_nodes_amount: " << output_nodes_amount << '\n';
        std::cout << "biases: " << '\n';
        for (uint32_t out = 0; out < output_nodes_amount; out++)
        {
            std::cout << biases[out] << ",  ";
        }
        std::cout << '\n'
                  << '\n';

        std::cout << "weights: " << '\n';
        for (uint32_t in = 0; in < input_nodes_amount; in++)
        {
            std::cout << "from input " << in << ": " << '\n';
            for (uint32_t out = 0; out < output_nodes_amount; out++)
            {
                std::cout << weights[in][out] << ", ";
            }
            std::cout << '\n';
        }

        std::cout << "bias gradient: " << '\n';
        for (uint32_t out = 0; out < output_nodes_amount; out++)
        {
            std::cout << bias_cost_gradient[out] << ",  ";
        }
        std::cout << '\n'
                  << '\n';
        std::cout << "weight gradient: " << '\n';
        for (uint32_t in = 0; in < input_nodes_amount; in++)
        {
            std::cout << "from input " << in << ": " << '\n';
            for (uint32_t out = 0; out < output_nodes_amount; out++)
            {
                std::cout << weight_cost_gradient[in][out] << ", ";
            }
            std::cout << '\n';
        }
        std::cout << '\n'
                  << '\n';
    }
    void output_layer(std::ofstream& file)
    {
        file << '\n'
             << '\n';
        file << "input_nodes_amount: " << input_nodes_amount << '\n';
        file << "output_nodes_amount: " << output_nodes_amount << '\n';
        file << "biases: " << '\n';
        for (uint32_t out = 0; out < output_nodes_amount; out++)
        {
            file << biases[out] << ",  ";
        }
        file << '\n'
             << '\n';

        file << "weights (1st row = from first input node to output node x ect.): " << '\n';
        for (uint32_t in = 0; in < input_nodes_amount; in++)
        {
            for (uint32_t out = 0; out < output_nodes_amount; out++)
            {
                file << weights[in][out] << ", ";
            }
            file << '\n';
        }

        file << '\n'
             << '\n';
        file << "bias gradient: " << '\n';
        for (uint32_t out = 0; out < output_nodes_amount; out++)
        {
            file << bias_cost_gradient[out] << ",  ";
        }
        file << '\n'
             << '\n';
        file << "weight gradient: " << '\n';
        for (uint32_t in = 0; in < input_nodes_amount; in++)
        {
            for (uint32_t out = 0; out < output_nodes_amount; out++)
            {
                file << weight_cost_gradient[in][out] << ", ";
            }
            file << '\n';
        }
        file << '\n'
             << std::endl;
    }
};
