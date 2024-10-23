#include "neural_network.hpp"

class Activation_function
{
public:
    long double activate(double &value);
    long double derivate(double &value);
};

// doesn't work?
class sigmoid_activation : public Activation_function
{
public:
    long double activate(long double &value)
    {
        // tanh from math.h is also an option
        return 1 / (1 + exp(-(double)value));
    }
    long double derivate(long double &value)
    {
        long double a = activate(value);
        return a * (1 - a);
    }
};
class hard_sigmoid_activation : public Activation_function
{
public:
    long double activate(long double &value)
    {
        if (value > 3)
        {
            return 1;
        }
        if (value < -3)
        {
            return 0;
        }
        return value * 0.166 + 0.5;
    }
    long double derivate(long double &value)
    {
        if (value > 3 || value < -3)
        {
            return 0;
        }
        return 0.166;
    }
};
class relu_activation : public Activation_function
{
public:
    long double activate(long double &value)
    {
        if (value < 0)
        {
            return 0;
        }
        return value;
    }
    long double derivate(long double &value)
    {
        if (value < 0)
        {
            return 0;
        }
        return 1;
    }
};
class leaky_relu_activation : public Activation_function
{
public:
    long double activate(long double &value)
    {
        if (value < 0)
        {
            return 0.1 * value;
        }
        return value;
    }
    long double derivate(long double &value)
    {
        if (value < 0)
        {
            return 0.1;
        }
        return 1;
    }
};
long double node_cost_function(std::vector<long double> predictedOutputs, std::vector<long double> expectedOutputs)
{
    // cost is sum (for all x,y pairs) of: 0.5 * (x-y)^2
    long double cost = 0;
    for (uint32_t i = 0; i < predictedOutputs.size(); i++)
    {
        long double error = predictedOutputs[i] - expectedOutputs[i];
        cost += error * error;
    }
    return 0.5 * cost;
}

long double node_cost_derivative(long double predictedOutput, long double expectedOutput)
{
    return predictedOutput - expectedOutput;
}
