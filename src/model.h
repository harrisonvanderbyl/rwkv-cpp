#include <torch/script.h>
#include <torch/library.h>
#include <torch/torch.h>
#include "block.h"
class RWKV : public torch::nn::Module
{
public:
    RWKV(int dims, int layers, int headsize)
    {
        std::cout << "Legacy RWKV" << std::endl;
        head = torch::nn::Linear(dims, headsize);
        emb = torch::nn::Embedding(headsize, dims);
        ln_out = torch::nn::LayerNorm(torch::nn::LayerNormOptions({dims}));
        ln_in = torch::nn::LayerNorm(torch::nn::LayerNormOptions({dims}));
        for (int i = 0; i < layers; i++)
        {
            blocks.push_back(Block(dims));
        }
    }
    RWKV(std::string path)
    {
        torch::jit::script::Module w = torch::jit::load("/home/harrison/projects/rwkv-cpp/model.pt");

        head = torch::nn::Linear(w.attr("head.weight").toTensor().sizes()[1], w.attr("head.weight").toTensor().sizes()[0]);
        head->weight = w.attr("head.weight").toTensor();
        ln_in = torch::nn::LayerNorm(torch::nn::LayerNormOptions({w.attr("blocks.0.ln0.bias").toTensor().sizes()[0]}));
        ln_in->bias = w.attr("blocks.0.ln0.bias").toTensor();
        ln_in->weight = w.attr("blocks.0.ln0.weight").toTensor();
        ln_out = torch::nn::LayerNorm(torch::nn::LayerNormOptions({w.attr("ln_out.weight").toTensor().sizes()[0]}));
        ln_out->weight = w.attr("ln_out.weight").toTensor();
        ln_out->bias = w.attr("ln_out.bias").toTensor();
        emb = torch::nn::Embedding(w.attr("emb.weight").toTensor().sizes()[0], w.attr("emb.weight").toTensor().sizes()[1]);
        emb->weight = w.attr("emb.weight").toTensor();

        for (int i = 0; i < 100; i++)
        {
            if (w.hasattr("blocks." + std::to_string(i) + ".ln1.bias"))
            {
                blocks.push_back(Block(i, w));
            }
            else
            {
                break;
            }
        }

        emptyState = w.attr("emptyState").toTensor();
    }
    torch::Tensor forward(torch::Tensor x, torch::Tensor state)
    {
        x = emb(x);
        x = ln_in(x);
        std::cout << state.sizes() << std::endl;
        for (int i = 0; i < blocks.size(); i++)
        {
            torch::Tensor rstate;
            std::tie(x, rstate) = blocks[i].forward(x, state[i]);
            state[i] = rstate;
        }
        x = ln_out(x);
        torch::Tensor outx = head(x);
        return outx[-1], state;
    }

    torch::Tensor emptyState;

private:
    torch::nn::Linear head = nullptr;
    torch::nn::Embedding emb = nullptr;
    torch::nn::LayerNorm ln_out = nullptr;
    torch::nn::LayerNorm ln_in = nullptr;

    std::vector<Block> blocks = {};
};
