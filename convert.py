import torch


class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])
        dims = my_values["blocks.0.ln0.bias"].shape[0]
        layers = len(list(filter(lambda k: k.startswith(
            "blocks.") and k.endswith(".ln1.bias"), my_values.keys())))

        print("dims", dims)
        print("layers", layers)

        emptyState = torch.zeros(layers, 5, dims)

        setattr(self, "emptyState", emptyState)


my_values = torch.load("model.pth", map_location="cpu")

# Save arbitrary values supported by TorchScript
# https://pytorch.org/docs/master/jit.html#supported-type
container = torch.jit.script(Container(my_values))
container.save("model.pt")
