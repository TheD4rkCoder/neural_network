#include "neural_network.hpp"

class Activation_function
{
public:
    virtual double activate(double &value);
    virtual double derivate(double &value);
};

class sigmoid_activation : public Activation_function
{
public:
    double activate(double &value)
    {
        // tanh from math.h is also an option
        return 1 / (1 + exp(-value));
    }
    double derivate(double &value)
    {
        double a = activate(value);
        return a * (1 - a);
    }
};
class hard_sigmoid_activation : public Activation_function
{
public:
    double activate(double &value)
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
    double derivate(double &value)
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
    double activate(double &value)
    {
        if (value < 0)
        {
            return 0;
        }
        return value;
    }
    double derivate(double &value)
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
    double activate(double &value)
    {
        if (value < 0)
        {
            return 0.1 * value;
        }
        return value;
    }
    double derivate(double &value)
    {
        if (value < 0)
        {
            return 0.1;
        }
        return 1;
    }
};
double node_cost_function(std::vector<double> predictedOutputs, std::vector<double> expectedOutputs)
{
    // cost is sum (for all x,y pairs) of: 0.5 * (x-y)^2
    double cost = 0;
    for (int i = 0; i < predictedOutputs.size(); i++)
    {
        double error = predictedOutputs[i] - expectedOutputs[i];
        cost += error * error;
    }
    return 0.5 * cost;
}

double node_cost_derivative(double predictedOutput, double expectedOutput)
{
    return predictedOutput - expectedOutput;
}
