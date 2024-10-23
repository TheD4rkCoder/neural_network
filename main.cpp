#pragma GCC optimize("Ofast")
#include "neural_network.hpp"

#define TRAINING_DATA_AMOUNT 1024
#define INPUT_NODES_AMOUNT 1
std::vector<TrainingDataPoint> training_data;

void generage_training_data()
{
    std::ofstream training_data_file("output/training_data.csv");

    training_data.clear();
    for (int i = 0; i < TRAINING_DATA_AMOUNT; ++i)
    {
        long double rand = random_ldouble(-3, 3);
        long double expected_output = std::sin(rand);
        training_data_file << rand << ", " << expected_output << std::endl;

        training_data.push_back(TrainingDataPoint({rand}, {expected_output}));
    }
    training_data_file.close();
}
void save_output(Network *n)
{
    std::ofstream out_file("output/output.csv");
    for (long double i = -3; i < 3; i += 0.01)
    {
        out_file << i << ", " << n->calculate({i})[0] << std::endl;
    }
    out_file.close();
    std::cout << "saved output" << std::endl
              << std::endl;
    n->output_network("output/network_after.csv");
}
void interruptable_training(bool *loop_condition, std::mutex *mutex, bool new_data, Network *n)
{
    int i = 0;
    long double average_cost;
    mutex->lock();
    while (*loop_condition)
    {
        mutex->unlock();
        if (new_data)
            generage_training_data();
        average_cost = n->train(training_data); // TODO: return average cost
        // if (i % 10 == 0)
        std::cout << "iteration: " << i++ << " cost: " << average_cost << std::endl;

        mutex->lock();
    }
    mutex->unlock();
}

int main()
{
    Network n(INPUT_NODES_AMOUNT, {3, 1});

    generage_training_data();
    std::cout << "average cost of training material: " << n.average_cost_of_training_data(training_data) << std::endl
              << std::endl;

    n.output_network("output/network_before.csv");

    std::string input;
    while (true)
    {
        std::cout << "g: generate new training data" << std::endl
                  << "c: run new training cycles with the same data" << std::endl
                  << "cg: run new training cycles with newly generated data" << std::endl
                  << "i: calculate manual input" << std::endl
                  << "e: edit learn rate" << std::endl
                  << "s: save output to output.csv" << std::endl
                  << "q: quit" << std::endl
                  << std::endl;
        std::cin >> input;
        if (input == "q")
        {
            break;
        }
        else if (input == "i")
        {
            std::vector<long double> num(INPUT_NODES_AMOUNT);
            for (uint32_t i = 0; i < num.size(); i++)
            {
                std::cout << "enter a double" << std::endl;
                std::cin >> input;
                try
                {
                    num[i] = std::stold(input);
                }
                catch (const std::invalid_argument &e)
                {
                    std::cerr << "Error: String '" << input << "' is not a valid double." << std::endl;
                }
            }
            std::cout << "output of the network: " << n.calculate(num)[0] << std::endl
                      << std::endl;
        }
        else if (input == "e")
        {
            std::cout << "What do you want to change the leanring rate to?" << std::endl;
            std::cin >> input;
            double num;
            try
            {
                num = std::stod(input);
                parameters.initial_learning_rate = num;
            }
            catch (const std::invalid_argument &e)
            {
                std::cerr << "Error: String '" << input << "' is not a valid double." << std::endl;
            }
        }
        else if (input == "g")
        {
            generage_training_data();
            std::cout << "average cost of new training material: " << n.average_cost_of_training_data(training_data) << std::endl;
            std::cout << "done generating training data" << std::endl
                      << std::endl;
        }
        else if (input == "c" || input == "cg")
        {
            bool new_data = false;
            if (input == "cg")
                new_data = true;
            std::cout << "How many training cycles?" << std::endl;
            std::cin >> input;
            int num = 0;
            try
            {
                num = std::stoi(input);
            }
            catch (const std::invalid_argument &e)
            {
                std::cerr << "Error: String '" << input << "' is not a valid integer." << std::endl;
            }

            if (num == -1)
            {
                bool loop_condition = true;
                std::mutex mutex;
                std::thread t(interruptable_training, &loop_condition, &mutex, new_data, &n);
                std::getchar();
                while (loop_condition)
                {
                    std::cout << "Press Enter to stop" << std::endl
                              << std::endl;
                    std::getchar();

                    mutex.lock();
                    loop_condition = false;
                    mutex.unlock();
                }
                t.join();
            }
            else
            {
                for (int i = 0; i < num; ++i)
                {
                    if (new_data)
                        generage_training_data();
                    long double cost = n.train(training_data);
                    // if (i % 10 == 0)
                    std::cout << "iteration: " << i << " cost: " << cost << std::endl;
                }
            }
            std::cout << "average cost of training material (after training): " << n.average_cost_of_training_data(training_data) << std::endl;
            std::cout << "learn rate: " << parameters.initial_learning_rate << std::endl;
            std::cout << "done training" << std::endl
                      << std::endl;
        }
        else if (input == "s")
        {
            save_output(&n);
        }
    }
}