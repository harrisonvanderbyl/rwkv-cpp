// Your First C++ Program
#include <torch/script.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <iostream>
#define PRINT(x) std::cout << x.sizes() << std::endl
/*
class Block(torch.nn.Module):
    def __init__(self, dims):
        super(Block, self).__init__()

        self.ln1 = torch.nn.LayerNorm(
            (dims,), device=device, dtype=dtype)
        self.ln2 = torch.nn.LayerNorm(
            dims, device=device, dtype=dtype)

        self.attkey = torch.nn.Linear(
            dims, dims, device=device)

        self.attvalue = torch.nn.Linear(
            dims, dims, device=device)

        self.attreceptance = torch.nn.Linear(
            dims, dims, device=device)

        self.atttime_mix_k = torch.nn.Parameter(
            torch.zeros(dims))
        self.atttime_mix_v = torch.nn.Parameter(
            torch.zeros(dims))
        self.atttime_mix_r = torch.nn.Parameter(
            torch.zeros(dims))

        self.time_first = torch.nn.Parameter(
            torch.zeros(dims))

        self.time_decay = torch.nn.Parameter(
            torch.zeros(dims))

        self.ffntime_mix_k = torch.nn.Parameter(
            torch.zeros(dims))
        self.ffntime_mix_r = torch.nn.Parameter(
            torch.zeros(dims))

        self.ffnkey = torch.nn.Linear(
            dims, dims*4, device=device)

        self.ffnvalue = torch.nn.Linear(
            dims*4, dims, device=device)

        self.ffnreceptance = torch.nn.Linear(
            dims, dims, device=device)

        self.attout = torch.nn.Linear(
            dims, dims, device=device)

    def loadFromBlinkDLCheckpoint(self, w, i):
        self.ln1.weight = torch.nn.Parameter(w[f"blocks.{i}.ln1.weight"])
        self.ln1.bias = torch.nn.Parameter(w[f"blocks.{i}.ln1.bias"])
        self.ln2.weight = torch.nn.Parameter(w[f"blocks.{i}.ln2.weight"])
        self.ln2.bias = torch.nn.Parameter(w[f"blocks.{i}.ln2.bias"])
        self.attkey.weight = torch.nn.Parameter(
            w[f"blocks.{i}.att.key.weight"])
        self.attvalue.weight = torch.nn.Parameter(
            w[f"blocks.{i}.att.value.weight"])
        self.ffnkey.weight = torch.nn.Parameter(
            w[f"blocks.{i}.ffn.key.weight"])
        self.ffnvalue.weight = torch.nn.Parameter(
            w[f"blocks.{i}.ffn.value.weight"])
        self.attout.weight = torch.nn.Parameter(
            w[f"blocks.{i}.att.output.weight"])
        self.ffnreceptance.weight = torch.nn.Parameter(
            w[f"blocks.{i}.ffn.receptance.weight"])
        self.attreceptance.weight = torch.nn.Parameter(
            w[f"blocks.{i}.att.receptance.weight"])

        self.atttime_mix_k = torch.nn.Parameter(
            w[f"blocks.{i}.att.time_mix_k"].squeeze())
        self.atttime_mix_v = torch.nn.Parameter(
            w[f"blocks.{i}.att.time_mix_v"].squeeze())
        self.atttime_mix_r = torch.nn.Parameter(
            w[f"blocks.{i}.att.time_mix_r"].squeeze())

        self.time_first = torch.nn.Parameter(
            w[f"blocks.{i}.att.time_first"].squeeze())

        self.time_decay = torch.nn.Parameter(
            w[f"blocks.{i}.att.time_decay"].squeeze().double().exp().neg().float())

        self.ffntime_mix_k = torch.nn.Parameter(
            w[f"blocks.{i}.ffn.time_mix_k"].squeeze())
        self.ffntime_mix_r = torch.nn.Parameter(
            w[f"blocks.{i}.ffn.time_mix_r"].squeeze())
        self.attreceptance.weight = torch.nn.Parameter(
            w[f"blocks.{i}.att.receptance.weight"])

    def processLayerx(self, k, v, rz: List[torch.Tensor], state, i: int):
        ww = self.time_first + k[i]
        p = torch.maximum(state[4], ww)

        e1 = (state[4] - p).exp()

        e2 = (ww - p).exp()

        a = e1 * (state[2]) + e2 * v[i]

        b = e1 * (state[3]) + e2

        wwn = state[4] + self.time_decay

        p1 = torch.maximum(wwn, k[i])

        e11 = (wwn - p1).exp()

        e21 = (k[i] - p1).exp()

        outb = e11 * state[2] + e21 * v[i]

        outc = e11 * state[3] + e21

        state[2:5] = torch.stack((outb, outc, p1))

        wkv = a / b

        rz.append(wkv)
        return rz, state

    # def processLayer(self, k, v, rz: List[torch.Tensor], state, xx: int, i: int): Mathematically identical
    #     ki = self.exp(k[i])
    #     wrd = self.divide(
    #         self.add(state[2], self.multiply(self.multiply(ki, v[i]), self.exp(self.time_first[xx]))), self.add(state[3], self.multiply(ki, self.exp(self.time_first[xx]))))

    #     state = self.scatter(state, self.scatterindices[1], self.multiply(self.exp(self.time_decay[xx]), self.add(
    #         state[2:4], self.stack((self.multiply(
    #             v[i], ki), ki)))))

    #     rz = self.arrayPush(rz, wrd, i)
    #     return rz, state

    def forward(self, x, state):

        xy = self.ln1(x)

        tc = xy.roll(1, 0)
        rmc = xy[-1]
        tc[0] = state[0]
        state[0] = rmc

        km = torch.lerp(tc, xy, self.atttime_mix_k)

        k = self.attkey(km)

        vm = torch.lerp(tc, xy, self.atttime_mix_v)

        v = self.attvalue(vm)

        rm = torch.lerp(tc, xy, self.atttime_mix_r)

        r = self.attreceptance(rm).sigmoid()

        rz = []

        for i in range(len(k)):
            rz, state = self.processLayerx(k, v, rz, state, i)

        rz = self.attout(torch.stack(rz)*r) + x

        ddd = self.ln2(rz)

        rc = ddd.roll(1, 0)
        dc = ddd[-1]
        rc[0] = state[1]
        state[1] = dc

        kmr = torch.lerp(rc, ddd, self.ffntime_mix_k)

        kf = self.ffnkey(kmr).relu()

        rmr = torch.lerp(rc, ddd, self.ffntime_mix_r)

        rf = self.ffnreceptance(rmr).sigmoid()

        rvm = self.ffnvalue(torch.square(kf))

        out = rvm * rf + rz

        return out, state

        # stuff


*/

// Now in c++

class Block : public torch::nn::Module
{

private:
    torch::nn::LayerNorm ln1 = nullptr;
    torch::nn::LayerNorm ln2 = nullptr;
    torch::nn::Linear attkey = nullptr;
    torch::nn::Linear attvalue = nullptr;
    torch::nn::Linear attreceptance = nullptr;
    torch::nn::Linear attout = nullptr;
    torch::nn::Linear ffnkey = nullptr;
    torch::nn::Linear ffnvalue = nullptr;
    torch::nn::Linear ffnreceptance = nullptr;
    torch::Tensor time_first;
    torch::Tensor time_decay;
    torch::Tensor atttime_mix_k;
    torch::Tensor atttime_mix_v;
    torch::Tensor atttime_mix_r;
    torch::Tensor ffntime_mix_k;
    torch::Tensor ffntime_mix_r;

public:
    Block(int dims)
    {

        ln1 = torch::nn::LayerNorm(torch::nn::LayerNormOptions({dims}));
        ln2 = torch::nn::LayerNorm(torch::nn::LayerNormOptions({dims}));
        attkey = torch::nn::Linear(dims, dims);
        attvalue = torch::nn::Linear(dims, dims);
        attreceptance = torch::nn::Linear(dims, dims);
        attout = torch::nn::Linear(dims, dims);
        ffnkey = torch::nn::Linear(dims, dims * 4);
        ffnvalue = torch::nn::Linear(dims * 4, dims);
        ffnreceptance = torch::nn::Linear(dims, dims);
        time_first = torch::zeros({dims});
        time_decay = torch::zeros({dims});
        atttime_mix_k = torch::zeros({dims});
        atttime_mix_v = torch::zeros({dims});
        atttime_mix_r = torch::zeros({dims});
        ffntime_mix_k = torch::zeros({dims});
        ffntime_mix_r = torch::zeros({dims});
    }

    Block(int i, torch::jit::script::Module w)

    {
        int dims = w.attr("blocks." + std::to_string(i) + ".att.key.weight").toTensor().size(0);
        ln1 = torch::nn::LayerNorm(torch::nn::LayerNormOptions({dims}));
        ln2 = torch::nn::LayerNorm(torch::nn::LayerNormOptions({dims}));
        attkey = torch::nn::Linear(dims, dims);
        attvalue = torch::nn::Linear(dims, dims);
        attreceptance = torch::nn::Linear(dims, dims);
        attout = torch::nn::Linear(dims, dims);
        ffnkey = torch::nn::Linear(dims, dims * 4);
        ffnvalue = torch::nn::Linear(dims * 4, dims);
        ffnreceptance = torch::nn::Linear(dims, dims);
        time_first = w.attr("blocks." + std::to_string(i) + ".att.time_first").toTensor().squeeze();
        time_decay = w.attr("blocks." + std::to_string(i) + ".att.time_decay").toTensor().squeeze().exp().neg();
        atttime_mix_k = w.attr("blocks." + std::to_string(i) + ".att.time_mix_k").toTensor().squeeze();
        atttime_mix_v = w.attr("blocks." + std::to_string(i) + ".att.time_mix_v").toTensor().squeeze();
        atttime_mix_r = w.attr("blocks." + std::to_string(i) + ".att.time_mix_r").toTensor().squeeze();
        ffntime_mix_k = w.attr("blocks." + std::to_string(i) + ".ffn.time_mix_k").toTensor().squeeze();
        ffntime_mix_r = w.attr("blocks." + std::to_string(i) + ".ffn.time_mix_r").toTensor().squeeze();

        ln1->weight = w.attr("blocks." + std::to_string(i) + ".ln1.weight").toTensor().squeeze();
        ln1->bias = w.attr("blocks." + std::to_string(i) + ".ln1.bias").toTensor().squeeze();
        ln2->weight = w.attr("blocks." + std::to_string(i) + ".ln2.weight").toTensor().squeeze();
        ln2->bias = w.attr("blocks." + std::to_string(i) + ".ln2.bias").toTensor().squeeze();
        attkey->weight = w.attr("blocks." + std::to_string(i) + ".att.key.weight").toTensor().squeeze();
        attvalue->weight = w.attr("blocks." + std::to_string(i) + ".att.value.weight").toTensor().squeeze();
        attreceptance->weight = w.attr("blocks." + std::to_string(i) + ".att.receptance.weight").toTensor().squeeze();
        attout->weight = w.attr("blocks." + std::to_string(i) + ".att.output.weight").toTensor().squeeze();
        ffnkey->weight = w.attr("blocks." + std::to_string(i) + ".ffn.key.weight").toTensor().squeeze();
        ffnvalue->weight = w.attr("blocks." + std::to_string(i) + ".ffn.value.weight").toTensor().squeeze();
        ffnreceptance->weight = w.attr("blocks." + std::to_string(i) + ".ffn.receptance.weight").toTensor().squeeze();
    }

    std::vector<torch::Tensor> processLayer(torch::Tensor k, torch::Tensor v, std::vector<torch::Tensor> rz, torch::Tensor state)
    {

        torch::Tensor ww = time_first + k;
        torch::Tensor p = torch::maximum(state[4], ww);

        torch::Tensor e1 = (state[4] - p).exp();

        torch::Tensor e2 = (ww - p).exp();

        torch::Tensor a = e1 * (state[2]) + e2 * v;

        torch::Tensor b = e1 * (state[3]) + e2;

        torch::Tensor wwn = state[4] + time_decay;

        torch::Tensor p1 = torch::maximum(wwn, k);

        torch::Tensor e11 = (wwn - p1).exp();

        torch::Tensor e21 = (k - p1).exp();

        torch::Tensor outb = e11 * state[2] + e21 * v;

        torch::Tensor outc = e11 * state[3] + e21;

        state[2] = outb;
        state[3] = outc;
        state[4] = p1;

        torch::Tensor wkv = a / b;

        rz.push_back(wkv);
        return rz;
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor state)
    {
        torch::Tensor xy = ln1(x);

        torch::Tensor tc = xy.roll(1, 0);
        torch::Tensor rmc = xy[-1];
        std::cout << tc.sizes() << std::endl;
        std::cout << state.sizes() << std::endl;
        tc[0] = state[0];
        state[0] = rmc;

        torch::Tensor km = torch::lerp(tc, xy, atttime_mix_k);

        torch::Tensor k = attkey(km);

        torch::Tensor vm = torch::lerp(tc, xy, atttime_mix_v);

        torch::Tensor v = attvalue(vm);

        torch::Tensor rm = torch::lerp(tc, xy, atttime_mix_r);

        torch::Tensor r = attreceptance(rm).sigmoid();

        std::vector<torch::Tensor> rz;

        for (int i = 0; i < k.size(0); i++)
        {
            rz = processLayer(k[i].squeeze(), v[i].squeeze(), rz, state);
        }

        torch::Tensor rz2 = torch::stack(rz);
        torch::Tensor rzz = attout(rz2 * r) + x;

        torch::Tensor ddd = ln2(rzz);

        torch::Tensor rc = ddd.roll(1, 0);
        torch::Tensor dc = ddd[-1];
        rc[0] = state[1];
        state[1] = dc;

        torch::Tensor kmr = torch::lerp(rc, ddd, ffntime_mix_k);

        torch::Tensor kf = ffnkey(kmr).relu();

        torch::Tensor rmr = torch::lerp(rc, ddd, ffntime_mix_r);

        torch::Tensor rf = ffnreceptance(rmr).sigmoid();

        torch::Tensor rvm = ffnvalue(torch::square(kf));

        torch::Tensor out = rvm * rf + rzz;

        return std::make_tuple(x, state);
    }
};