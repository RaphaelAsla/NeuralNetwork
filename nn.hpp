#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

/* initialize the generator for the uniform_real_distribution */
static std::mt19937 gen(std::random_device{}());

/* activation function */
inline double sigmoid(const double x) {
    return 1 / (1 + exp(-x));
}

/* derivative of the activation function */
inline double sigmoid_derivative(const double x) {
    return x * (1 - x);
}

struct Neuron {
    std::vector<double> weights;
    double output;
    double error;
    double bias;

    Neuron() = default;

    /* initialize the weights and biases */
    Neuron(const int input_neurons_count) {
        weights.reserve(input_neurons_count);
        for (int i = 0; i < input_neurons_count; i++) {
            weights.emplace_back(std::uniform_real_distribution<double>{0, 1}(gen));
        }
        bias = std::uniform_real_distribution<double>{0, 1}(gen);
    }
};

struct Layer {
    std::vector<Neuron> neurons;

    Layer() = default;

    /* initialize the neurons of the layer */
    Layer(const int input_neurons_count, const int neurons_count) {
        neurons.reserve(neurons_count);
        for (int i = 0; i < neurons_count; i++) {
            neurons.emplace_back(Neuron(input_neurons_count));
        }
    }
};

struct Network {
    std::vector<Layer> layers;
    double learning_rate;

    Network() = default;

    /* initialize from file */
    Network(const std::string& path) {
        std::ifstream file(path, std::ios::in | std::ios::binary);

        if (!file.is_open()) {
            std::cerr << "Error, could not open file for loading\n";
            exit(0);
        }

        /* initialize the learnign rate */
        file.read(reinterpret_cast<char*>(&learning_rate), sizeof(double));

        /* initialize the number of layers in the Network */
        int layers_count;
        file.read(reinterpret_cast<char*>(&layers_count), sizeof(int));
        layers.reserve(layers_count);

        for (int i = 0; i < layers_count; i++) {
            /* initialize the number of neurons per layer */
            Layer layer;
            int neurons_count;
            file.read(reinterpret_cast<char*>(&neurons_count), sizeof(int));
            layer.neurons.reserve(neurons_count);
            for (int j = 0; j < neurons_count; j++) {
                /* initialize the number of weights per neuron */
                Neuron neuron;
                int weights_count;
                file.read(reinterpret_cast<char*>(&weights_count), sizeof(int));
                neuron.weights.resize(weights_count);
                /* initialize the weights and biases for each neuron */
                file.read(reinterpret_cast<char*>(neuron.weights.data()), sizeof(double) * weights_count);
                file.read(reinterpret_cast<char*>(&neuron.bias), sizeof(double));
                layer.neurons.emplace_back(std::move(neuron));
            }
            layers.emplace_back(std::move(layer));
        }
        file.close();
    }

    /* initialize from a given topology */
    Network(const std::vector<int> topology, const double rate) {
        learning_rate = rate;
        layers.reserve(topology.size() - 1);
        /* outputs of the first layer are our inputs values, so we can just skip the first layer */
        for (int i = 0; i < topology.size() - 1; i++) {
            layers.emplace_back(Layer(topology[i], topology[i + 1]));
        }
    }

    /* forward propagation */
    std::vector<double> Predict(std::vector<double> inputs) {
        std::vector<double> outputs;
        for (auto& layer : layers) {
            /* clear so we can store the current layer's output values */
            outputs.clear();
            for (auto& neuron : layer.neurons) {
                int k = 0;
                double sum = neuron.bias;
                for (auto& weight : neuron.weights) {
                    sum += weight * inputs[k];
                    k++;
                }
                /* activate neuron */
                neuron.output = sigmoid(sum);
                /* save the output of each layer */
                outputs.emplace_back(neuron.output);
            }
            /* use the outputs of the current layer as inputs for the next */
            inputs = outputs;
        }
        /* return the last layer's output values */
        return outputs;
    }

    /* backward propagation */
    void Train(std::vector<double> inputs, const std::vector<double> targets) {
        if (targets.size() != layers.back().neurons.size()) {
            throw std::invalid_argument("The number of output neurons must be equal with the number of targets\n");
        }

        /* make a prediction to then track the errors */
        Predict(inputs);

        /* compute the output errors for the last layer */
        int k = 0;
        for (auto& neuron : layers.back().neurons) {
            neuron.error = sigmoid_derivative(neuron.output) * (targets[k] - neuron.output);
            k++;
        }

        /* compute the errors for the hidden layers */
        for (int i = layers.size() - 2; i >= 0; i--) {
            for (int j = 0; j < layers[i].neurons.size(); j++) {
                /* sum of neurons errors from the next layer */
                double error = 0;
                for (int k = 0; k < layers[i + 1].neurons.size(); k++) {
                    error += layers[i + 1].neurons[k].error * layers[i + 1].neurons[k].weights[j];
                }
                layers[i].neurons[j].error = sigmoid_derivative(layers[i].neurons[j].output) * error;
            }
        }

        /* update the weights and biases */
        for (auto& layer : layers) {
            for (auto& neuron : layer.neurons) {
                neuron.bias += learning_rate * neuron.error;
                int k = 0;
                for (auto& weight : neuron.weights) {
                    weight += learning_rate * neuron.error * inputs[k];
                    k++;
                }
            }
            /* clear the inputs and get the outputs of the current layer to use as inputs for the next */
            inputs.clear();
            for (auto& neuron : layer.neurons) {
                inputs.emplace_back(neuron.output);
            }
        }
    }

    /* save Network to binary file */
    void Save(const std::string& path) {
        std::ofstream file(path, std::ios::out | std::ios::binary);

        if (!file.is_open()) {
            std::cerr << "Error, could not open file for saving\n";
            exit(0);
        }

        /* save the learnign rate */
        file.write(reinterpret_cast<const char*>(&learning_rate), sizeof(double));

        /* save the number of layers in the Network */
        int layers_count = layers.size();
        file.write(reinterpret_cast<const char*>(&layers_count), sizeof(int));

        for (auto& layer : layers) {
            /* save the number of neurons per layer */
            int neuronsCount = layer.neurons.size();
            file.write(reinterpret_cast<const char*>(&neuronsCount), sizeof(int));
            for (auto& neuron : layer.neurons) {
                /* save the number of weights per neuron */
                int weights_count = neuron.weights.size();
                file.write(reinterpret_cast<const char*>(&weights_count), sizeof(int));
                /* save the weights and biases for each neuron */
                file.write(reinterpret_cast<const char*>(neuron.weights.data()), sizeof(double) * weights_count);
                file.write(reinterpret_cast<const char*>(&neuron.bias), sizeof(double));
            }
        }
        file.close();
    }

    /* load Network from binary file */
    void Load(const std::string& path) {
        std::ifstream file(path, std::ios::in | std::ios::binary);

        if (!file.is_open()) {
            std::cerr << "Error, could not open file for loading\n";
            exit(0);
        }

        /* initialize the learnign rate */
        file.read(reinterpret_cast<char*>(&learning_rate), sizeof(double));

        /* initialize the number of layers in the Network */
        int layers_count;
        file.read(reinterpret_cast<char*>(&layers_count), sizeof(int));
        if (layers_count != layers.size()) {
            throw std::invalid_argument("Trying to initialize from neural network with different topology\n");
        }

        for (int i = 0; i < layers_count; i++) {
            /* initialize the number of neurons per layer */
            Layer& layer = layers[i];
            int neurons_count;
            file.read(reinterpret_cast<char*>(&neurons_count), sizeof(int));
            if (neurons_count != layer.neurons.size()) {
                throw std::invalid_argument("Trying to initialize from neural network with different topology\n");
            }
            for (int j = 0; j < neurons_count; j++) {
                /* initialize the number of weights per neuron */
                Neuron& neuron = layer.neurons[j];
                int weights_count;
                file.read(reinterpret_cast<char*>(&weights_count), sizeof(int));
                if (weights_count != neuron.weights.size()) {
                    throw std::invalid_argument("Trying to initialize from neural network with different topology\n");
                }
                /*  initialize the weights and biases for each neuron */
                file.read(reinterpret_cast<char*>(neuron.weights.data()), sizeof(double) * weights_count);
                file.read(reinterpret_cast<char*>(&neuron.bias), sizeof(double));
            }
        }
        file.close();
    }
};
