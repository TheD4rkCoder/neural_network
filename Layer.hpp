#include "neural_network.hpp"
class Layer
{
private:
    // Used for adding momentum to gradient descent:
    std::vector<std::vector<long double>> weight_velocities;
    std::vector<std::vector<long double>> bias_velocities;

    std::vector<long double> last_weighted_inputs;
    std::vector<long double> last_inputs;
    

    void initialize_with_random_weigths()
    {
        for (int out = 0; out < output_nodes_amount; ++out)
        {
            for (int in = 0; in < input_nodes_amount; ++in)
            {
                weights[in][out] = random_ldouble(-3, 3) / input_nodes_amount;
            }
        }
    }
    void update_gradients(std::vector<long double> node_derivatives)
    {
        for (int out = 0; out < output_nodes_amount; ++out)
        {
            for (int in = 0; in < input_nodes_amount; ++in)
            {
                weight_cost_gradient[in][out] = last_inputs[in] * node_derivatives[out];
            }
            bias_cost_gradient[out] = node_derivatives[out];
        }
    }

public:
    int output_nodes_amount;
    int input_nodes_amount;

    std::vector<std::vector<long double>> weights; // [from][to]
    std::vector<long double> biases;               // added to sum of weights

    std::vector<std::vector<long double>> weight_cost_gradient;
    std::vector<long double> bias_cost_gradient;

    Layer(int input_nodes_amount, int output_nodes_amount)
    {
        this->input_nodes_amount = input_nodes_amount;
        this->output_nodes_amount = output_nodes_amount;
        weights = std::vector<std::vector<long double>>(input_nodes_amount);
        biases = std::vector<long double>(output_nodes_amount);
        weight_cost_gradient = std::vector<std::vector<long double>>(input_nodes_amount);
        bias_cost_gradient = std::vector<long double>(output_nodes_amount);
        for (int i = 0; i < input_nodes_amount; ++i)
        {
            weights[i] = std::vector<long double>(output_nodes_amount);
            weight_cost_gradient[i] = std::vector<long double>(output_nodes_amount);
        }
        last_weighted_inputs = std::vector<long double>(output_nodes_amount);

        initialize_with_random_weigths();
    }
    std::vector<long double> calculate_layer_result(std::vector<long double> input)
    {
        std::vector<long double> output = std::vector<long double>(output_nodes_amount);
        for (int out = 0; out < output_nodes_amount; ++out)
        {
            last_weighted_inputs[out] = biases[out];
            for (int in = 0; in < input_nodes_amount; ++in)
            {
                if (!std::isinf(last_weighted_inputs[out] + weights[in][out] * input[in]))
                    last_weighted_inputs[out] += weights[in][out] * input[in]; // becomes nan here?
            }
            output[out] = parameters.activation.activate(last_weighted_inputs[out]);
        }
        last_inputs = input;
        return output;
    }
    void apply_gradient_descent()
    {
        for (int out = 0; out < output_nodes_amount; out++)
        {
            if (!std::isinf(biases[out] - bias_cost_gradient[out] * parameters.initial_learning_rate))
                biases[out] -= bias_cost_gradient[out] * parameters.initial_learning_rate;

            for (int in = 0; in < input_nodes_amount; in++)
            {
                if (!std::isinf(weights[in][out] - weight_cost_gradient[in][out] * parameters.initial_learning_rate))
                    weights[in][out] -= weight_cost_gradient[in][out] * parameters.initial_learning_rate;
            }
        }
    }
    std::vector<long double> back_propagation(std::vector<long double> out_node_derivatives)
    {
        std::vector<long double> in_node_derivatives = std::vector<long double>(input_nodes_amount, 0);

        for (int out = 0; out < output_nodes_amount; ++out)
        {
            if (!std::isinf(out_node_derivatives[out] * parameters.activation.derivate(last_weighted_inputs[out])))
                out_node_derivatives[out] *= parameters.activation.derivate(last_weighted_inputs[out]);

            for (int in = 0; in < input_nodes_amount; ++in)
            {
                if (!std::isinf(in_node_derivatives[in] + out_node_derivatives[out] * weights[in][out]))
                    in_node_derivatives[in] += out_node_derivatives[out] * weights[in][out];
            }
        }
        update_gradients(out_node_derivatives);
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
        for (int out = 0; out < output_nodes_amount; out++)
        {
            std::cout << biases[out] << ",  ";
        }
        std::cout << std::endl
                  << std::endl;

        std::cout << "weights: " << std::endl;
        for (int in = 0; in < input_nodes_amount; in++)
        {
            std::cout << "from input " << in << ": " << std::endl;
            for (int out = 0; out < output_nodes_amount; out++)
            {
                std::cout << weights[in][out] << ", ";
            }
            std::cout << std::endl;
        }

        std::cout << "bias gradient: " << std::endl;
        for (int out = 0; out < output_nodes_amount; out++)
        {
            std::cout << bias_cost_gradient[out] << ",  ";
        }
        std::cout << std::endl
                  << std::endl;
        std::cout << "weight gradient: " << std::endl;
        for (int in = 0; in < input_nodes_amount; in++)
        {
            std::cout << "from input " << in << ": " << std::endl;
            for (int out = 0; out < output_nodes_amount; out++)
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
        for (int out = 0; out < output_nodes_amount; out++)
        {
            file << biases[out] << ",  ";
        }
        file << std::endl
             << std::endl;

        file << "weights (1st row = from first input node to output node x ect.): " << std::endl;
        for (int in = 0; in < input_nodes_amount; in++)
        {
            for (int out = 0; out < output_nodes_amount; out++)
            {
                file << weights[in][out] << ", ";
            }
            file << std::endl;
        }

        file << std::endl
             << std::endl;
        file << "bias gradient: " << std::endl;
        for (int out = 0; out < output_nodes_amount; out++)
        {
            file << bias_cost_gradient[out] << ",  ";
        }
        file << std::endl
             << std::endl;
        file << "weight gradient: " << std::endl;
        for (int in = 0; in < input_nodes_amount; in++)
        {
            for (int out = 0; out < output_nodes_amount; out++)
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
