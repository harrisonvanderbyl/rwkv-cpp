// Your First C++ Program

#include <iostream>
#include "src/model.h"

// Convert to C++:

int main()
{
    // Launch file selector

    RWKV rwkv("/home/harrison/projects/rwkv-cpp/model.pt");

    std::cout << rwkv.forward(torch::zeros(2).to(torch::kInt32), rwkv.emptyState) << std::endl;
}