
#pragma once
#include <getopt.h>
#include <string>

namespace thunder {

enum class ReorderAlgo { None = 0, Sorting = 1 };

const char* reorder_algo_to_string(ReorderAlgo algo)
{
    switch (algo) {
        case ReorderAlgo::None:
            return "None";
        case ReorderAlgo::Sorting:
            return "Sorting";
        default:
            return "Unknown";
    }
}

struct Config {
    std::string input_file;
    std::string output_file;
    int         ncol_dense      = 32;
    int         exec_iterations = 100;
    bool        verify          = false;
    ReorderAlgo reorder         = ReorderAlgo::None;
};

std::string option_hints = "              [-i input_file]\n"
                           "              [-o output_file]\n"
                           "              [-c columns_of_dense_matrix\n"
                           "              [-e execution_iterations]\n"
                           "              [-v verify_results (1 or 0)]\n"
                           "              [-r reorder_algorithm (0: none, 1: row sorting)]\n";

auto program_options(int argc, char* argv[])
{
    Config config;
    int    opt;
    if (argc == 1) {
        printf("Usage: %s ... \n%s", argv[0], option_hints.c_str());
        std::exit(EXIT_FAILURE);
    }
    while ((opt = getopt(argc, argv, "e:r:i:c:o:v:")) != -1) {
        switch (opt) {
            case 'i':
                config.input_file = optarg;
                break;
            case 'o':
                config.output_file = optarg;
                break;
            case 'e':
                config.exec_iterations = std::stoi(optarg);
                break;
            case 'c':
                config.ncol_dense = std::stoi(optarg);
                break;
            case 'r':
                config.reorder = static_cast<ReorderAlgo>(std::stoi(optarg));
                break;
            case 'v':
                config.verify = std::stoi(optarg);
                break;
            default:
                printf("Usage: %s ... \n%s", argv[0], option_hints.c_str());
                exit(EXIT_FAILURE);
        }
    }
    if (config.input_file.empty()) {
        printf("The input path is not provided!\n");
        exit(EXIT_FAILURE);
    }
    printf("--------experimental setting--------\n");
    if (!config.input_file.empty()) {
        printf("input path: %s\n", config.input_file.c_str());
    }
    if (config.reorder != ReorderAlgo::None) {
        printf("reorder algorithm: %s\n", reorder_algo_to_string(config.reorder));
    }
    if (!config.output_file.empty()) {
        printf("output path: %s\n", config.output_file.c_str());
    }
    if (config.exec_iterations > 0) {
        printf("execution iterations: %d\n", config.exec_iterations);
    }
    printf("\n");
    return config;
}
}  // namespace thunder