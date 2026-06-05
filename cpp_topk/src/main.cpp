#include "src/types.h"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    try {
        return top_k::run_cli(argc, argv);
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
