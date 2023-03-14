// Your First C++ Program

#include <iostream>
#include "src/model.h"

// Convert to C++:
// Get path from args
int main(int argc, char *argv[])
{
    auto path = argv[1];
    torch::string dtype = "float32";
    try
    {
        dtype = torch::string(argv[2]);
    }
    catch (...)
    {
        std::cout << "No dtype specified, defaulting to float32" << std::endl;
    }
    torch::string runtimedtype = "float32";
    try
    {
        runtimedtype = torch::string(argv[3]);
    }
    catch (...)
    {
        std::cout << "No runtime dtype specified, defaulting to float32" << std::endl;
    }

    if (dtype != "float32" && dtype != "float64" && dtype != "bloat16")
    {
        std::cout << "Invalid dtype specified, defaulting to float32" << std::endl;
        dtype = "float32";
    }

    if (runtimedtype != "float32" && runtimedtype != "float64" && runtimedtype != "bloat16")
    {
        std::cout << "Invalid runtime dtype specified, defaulting to float32" << std::endl;
        runtimedtype = "float32";
    }

    // Convert to torch dtype
    auto torch_dtype = torch::kFloat32;
    auto torch_runtimedtype = torch::kFloat32;

    if (dtype == "float32")
    {
        torch_dtype = torch::kFloat32;
    }
    else if (dtype == "float64")
    {
        torch_dtype = torch::kFloat64;
    }
    else if (dtype == "bloat16")
    {
        torch_dtype = torch::kBFloat16;
    }
    else if (dtype == "float16")
    {
        torch_dtype = torch::kFloat16;
    }

    if (runtimedtype == "float32")
    {
        torch_runtimedtype = torch::kFloat32;
    }
    else if (runtimedtype == "float64")
    {
        torch_runtimedtype = torch::kFloat64;
    }
    else if (runtimedtype == "bloat16")
    {
        torch_runtimedtype = torch::kBFloat16;
    }
    std::cout << "dtype: " << dtype << std::endl;
    RWKV rwkv(path, torch_dtype, torch_runtimedtype);

    std::cout << rwkv.forward(torch::zeros(2).to(torch::kInt32), rwkv.emptyState) << std::endl;
}