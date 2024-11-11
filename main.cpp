#include <iomanip>
#include <iostream>

#include "nn.hpp"

int main() {
    Network Net({2, 3, 2}, 1);
    // Network Net("net.bin");

    /* "XOR" and "AND" gate training */
    /*             {XOR_out, AND_out}*/
    for (size_t i = 0; i < 100000; i++) {
        Net.Train({1, 1}, {0, 1});
        Net.Train({1, 0}, {1, 0});
        Net.Train({0, 1}, {1, 0});
        Net.Train({0, 0}, {0, 0});
    }

    std::cout << std::fixed << std::setprecision(10);

    /* XOR gate: output 1 if the two inputs are different, 0 otherwise */
    std::cout << "Testing XOR gate with input (1, 1). Expected output: 0\n";
    std::cout << "Output: " << Net.Predict({1, 1})[0] << "\n\n";

    std::cout << "Testing XOR gate with input (0, 1). Expected output: 1\n";
    std::cout << "Output: " << Net.Predict({0, 1})[0] << "\n\n";

    std::cout << "Testing XOR gate with input (1, 0). Expected output: 1\n";
    std::cout << "Output: " << Net.Predict({1, 0})[0] << "\n\n";

    std::cout << "Testing XOR gate with input (0, 0). Expected output: 0\n";
    std::cout << "Output: " << Net.Predict({0, 0})[0] << "\n\n";

    std::cout << "======================================================\n\n";

    /* AND gate: output 1 if the two inputs are equal, 0 otherwise */
    std::cout << "Testing AND gate with input (1, 1). Expected output: 1\n";
    std::cout << "Output: " << Net.Predict({1, 1})[1] << "\n\n";

    std::cout << "Testing AND gate with input (0, 1). Expected output: 0\n";
    std::cout << "Output: " << Net.Predict({0, 1})[1] << "\n\n";

    std::cout << "Testing AND gate with input (1, 0). Expected output: 0\n";
    std::cout << "Output: " << Net.Predict({1, 0})[1] << "\n\n";

    std::cout << "Testing AND gate with input (0, 0). Expected output: 0\n";
    std::cout << "Output: " << Net.Predict({0, 0})[1] << "\n\n";

    // Net.Save("net.bin");
}
