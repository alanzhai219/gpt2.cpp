#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include "csrc/modeling.hpp"
#include "csrc/tokenizer.hpp"
#include "csrc/weights.hpp"

namespace {

struct Args {
    std::string command;
    std::string prompt;
    std::string weights_dir = "weights";
    int max_tokens = 50;
    float temperature = 0.8F;
    int top_k = 40;
    uint64_t seed = 42;
};

void usage() {
    std::printf(
        "usage:\n"
        "  gpt2 generate \"<prompt>\" \"<weights_dir>\"\n"
        "  gpt2 generate \"<prompt>\" \"<weights_dir>\" Max-Tokens Temperature Top-K Seed"
        "\n");
}

void init_args(char* argv[], Args& arg) {
    // 0 : bin itself
    // 1 : command
    // 2 : prompt
    // 3 : weights_dir
    // 4 : max_tokens
    // 5 : temperature
    // 6 : top_k
    // 7 : seed
    arg.command = argv[1];
    arg.prompt = argv[2];
    arg.weights_dir = argv[3];
    arg.max_tokens = std::atoi(argv[4]);
    arg.temperature = std::atof(argv[5]);
    arg.top_k = std::atoi(argv[6]);
    arg.seed = std::atoi(argv[7]);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return EXIT_FAILURE;
    }
    Args args;
    init_args(argv, args);
    args.command = argv[1];

    try {
        const std::string model_path = args.weights_dir + "/model.safetensors";
        std::fprintf(stderr, "loading tokenizer...\n");
        tk::Tokenizer tokenizer(args.weights_dir + "/encoder.json", args.weights_dir + "/vocab.bpe");
        std::fprintf(stderr, "loading weights (%s)...\n", model_path.c_str());
        gpt2::GPT2 model(gpt2::build_gpt2_weights(gpt2::load_safetensors(model_path), {}));

        if (args.command == "generate") {
            const std::string output = model.generate(
                tokenizer, args.prompt, args.max_tokens,
                args.temperature, args.top_k, args.seed);
            std::printf("%s\n", output.c_str());
            return 0;
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
