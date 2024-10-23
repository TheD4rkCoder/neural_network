#include "neural_network.hpp"
class Layer
{
private:
    // Used for adding momentum to gradient descent (not used yet D: )
    std::vector<std::vector<long double>> weight_velocities;
    std::vector<std::vector<long double>> bias_velocities;

    std::vector<long double> last_weighted_inputs;
    std::vector<long double> last_inputs;

    void initialize_with_random_weigths()
    {
        for (uint32_t out = 0; out < output_nodes_amount; ++out)
        {
            for (uint32_t in = 0; in < input_nodes_amount; ++in)
            {
                weights[in][out] = random_ldouble(-3, 3) / input_nodes_amount;
            }
        }
    }

    void update_gradients(std::vector<long double> node_derivatives)
    {
        for (uint32_t out = 0; out < output_nodes_amount; ++out)
        {
            for (uint32_t in = 0; in < input_nodes_amount; ++in)
            {
                weight_cost_gradient[in][out] = last_inputs[in] * node_derivatives[out];
                // if (weight_cost_gradient[in][out] > 5)
                // {
                //     weight_cost_gradient[in][out] = 5;
                // }
                // else if (weight_cost_gradient[in][out] < -5)
                // {
                //     weight_cost_gradient[in][out] = -5;
                // }
            }
            bias_cost_gradient[out] = node_derivatives[out];
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
        weights = std::vector<std::vector<long double>>(input_nodes_amount);
        biases = std::vector<long double>(output_nodes_amount);
        weight_cost_gradient = std::vector<std::vector<long double>>(input_nodes_amount);
        bias_cost_gradient = std::vector<long double>(output_nodes_amount);
        for (uint32_t i = 0; i < input_nodes_amount; ++i)
        {
            weights[i] = std::vector<long double>(output_nodes_amount);
            weight_cost_gradient[i] = std::vector<long double>(output_nodes_amount);
        }
        last_weighted_inputs = std::vector<long double>(output_nodes_amount);

        initialize_with_random_weigths();
    }

    Layer(std::ifstream file)
    {
        // TODO
    }

    std::vector<long double> calculate_layer_result(std::vector<long double>& input)
    {
        std::vector<long double> output = std::vector<long double>(output_nodes_amount);
        for (uint32_t out = 0; out < output_nodes_amount; ++out)
        {
            last_weighted_inputs[out] = biases[out];
            for (uint32_t in = 0; in < input_nodes_amount; ++in)
            {
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
            biases[out] -= bias_cost_gradient[out] * parameters.initial_learning_rate;

            for (uint32_t in = 0; in < input_nodes_amount; in++)
            {
                weights[in][out] -= weight_cost_gradient[in][out] * parameters.initial_learning_rate;
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
        apply_gradient_descent();
        return in_node_derivatives;
    }

    void output_layer()
    {
        std::cout << std::endl
                  << std::endl;
        std::cout << "input_nodes_amount: " << input_nodes_amount << std::endl;
        std::cout << "output_nodes_amount: " << output_nodes_amount << std::endl;
        std::cout << "biases: " << std::endl;
        for (uint32_t out = 0; out < output_nodes_amount; out++)
        {
            std::cout << biases[out] << ",  ";
        }
        std::cout << std::endl
                  << std::endl;

        std::cout << "weights: " << std::endl;
        for (uint32_t in = 0; in < input_nodes_amount; in++)
        {
            std::cout << "from input " << in << ": " << std::endl;
            for (uint32_t out = 0; out < output_nodes_amount; out++)
            {
                std::cout << weights[in][out] << ", ";
            }
            std::cout << std::endl;
        }

        std::cout << "bias gradient: " << std::endl;
        for (uint32_t out = 0; out < output_nodes_amount; out++)
        {
            std::cout << bias_cost_gradient[out] << ",  ";
        }
        std::cout << std::endl
                  << std::endl;
        std::cout << "weight gradient: " << std::endl;
        for (uint32_t in = 0; in < input_nodes_amount; in++)
        {
            std::cout << "from input " << in << ": " << std::endl;
            for (uint32_t out = 0; out < output_nodes_amount; out++)
            {
                std::cout << weight_cost_gradient[in][out] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl
                  << std::endl;
    }
    void output_layer(std::string file_name)
    {
        std::ofstream file(file_name, std::ios::app);

        file << std::endl
             << std::endl;
        file << "input_nodes_amount: " << input_nodes_amount << std::endl;
        file << "output_nodes_amount: " << output_nodes_amount << std::endl;
        file << "biases: " << std::endl;
        for (uint32_t out = 0; out < output_nodes_amount; out++)
        {
            file << biases[out] << ",  ";
        }
        file << std::endl
             << std::endl;

        file << "weights (1st row = from first input node to output node x ect.): " << std::endl;
        for (uint32_t in = 0; in < input_nodes_amount; in++)
        {
            for (uint32_t out = 0; out < output_nodes_amount; out++)
            {
                file << weights[in][out] << ", ";
            }
            file << std::endl;
        }

        file << std::endl
             << std::endl;
        file << "bias gradient: " << std::endl;
        for (uint32_t out = 0; out < output_nodes_amount; out++)
        {
            file << bias_cost_gradient[out] << ",  ";
        }
        file << std::endl
             << std::endl;
        file << "weight gradient: " << std::endl;
        for (uint32_t in = 0; in < input_nodes_amount; in++)
        {
            for (uint32_t out = 0; out < output_nodes_amount; out++)
            {
                file << weight_cost_gradient[in][out] << ", ";
            }
            file << std::endl;
        }
        file << std::endl
             << std::endl;
        file.close();
    }
};
