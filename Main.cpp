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
    std::string beforecontext = std::string("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n");
    std::string instructionContext = std::string("###Instruction:\nplease write a long story using information and themes provided that is at least 100 words long. You can use markdown to format your text.\n\n");
    
    
    while (1){
    auto baseState = emptyState.clone();
   
    std::cout << "Enter Story Details: ";
    std::string inp = "";
    std::getline(std::cin, inp);

    
    std::string requestContext = std::string("###Input:\n" + inp + "\n\n");
    std::string aftercontext = std::string("\n###Response:\n");
	std::vector<int> context = tokenizer.encodeTokens(beforecontext + instructionContext + requestContext + aftercontext);
    at::Tensor tok;
    for (int i = 0; i < context.size(); i++)
    {
        tok = torch::tensor({context[i]}).to(torch::kInt32);
        auto out = RWKV.forward(std::vector<c10::IValue>{tok, baseState});
        baseState = out.toTuple()->elements()[1].toTensor();

    }
    
    auto currentState = baseState.clone();

    std::vector<int> moutputs = {tok.item<int>()};

    

    for (int i = 0; i < 500; i++)
    {
        auto out = RWKV.forward(std::vector<c10::IValue>{torch::reshape(tok,{1}), currentState});
        tok = out.toTuple()->elements()[0].toTensor();
        currentState = out.toTuple()->elements()[1].toTensor();
        float temp = 0.9;
        float top_p = 0.9;
        auto probs = torch::softmax(tok, -1);
        torch::Tensor sorted_probs;
        std::tie(sorted_probs, std::ignore) = torch::sort(probs, -1, true);
        auto cumulative_probs = torch::cumsum(sorted_probs, -1);
        cumulative_probs = cumulative_probs.masked_fill(cumulative_probs < top_p, 0.0);
        auto cutoff = sorted_probs[torch::argmax(cumulative_probs)];
        probs = probs.masked_fill(probs < cutoff, 0.0);
        probs = torch::pow(probs, 1.0 / temp);
        probs = probs / torch::sum(probs, -1);
        tok = torch::multinomial(probs, 1);
           

        tok = tok.cpu();
        moutputs.push_back(tok.item<int>());
        if (tok.item<int>() == 0)
        {
            break;
        }
        cout << tokenizer.decodeTokens({tok.item<int>()});
    }
    
    

    }

}