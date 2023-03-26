// Your First C++ Program

#include <iostream>
#include <torch/script.h>
#include <torch/library.h>
#include <torch/torch.h>
#include "./cudaops/wrapper.cpp"
#include "tokenizer.h"
#include <stdio.h>
// Convert to C++:
// Get path from args
int main(int argc, char *argv[])
{
    auto path = argv[1];
    
    auto RWKV = torch::jit::load(path);

    auto emptyState = torch::zeros({32,5,4096}).to(torch::kFloat64).cuda();
   
    auto B = torch::zeros(1).to(torch::kInt32);

    auto tokenizer = Tokenizer("20B_tokenizer.json");
    
    cout << tokenizer.encodeTokens("Hello World") << endl;
    cout << tokenizer.decodeTokens({12092, 3645}) << endl;

	// load tokenizer

	
    std::vector<int> moutputs = {};

    

    auto time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; i++)
    {
        auto out = RWKV.forward(std::vector<c10::IValue>{torch::reshape(B,{1}), emptyState});
        B = out.toTuple()->elements()[0].toTensor();
        emptyState = out.toTuple()->elements()[1].toTensor();

        B = torch::argmax(B.cpu());
        moutputs.push_back(B.item<int>());
    }
    
    cout << tokenizer.decodeTokens(moutputs) << endl;
    cout << moutputs << endl;


    auto time2 = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time).count() << "ms / 100 tokens" << std::endl;

}