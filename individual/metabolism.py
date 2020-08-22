import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

dtype = torch.float
device = torch.device("cuda:0")


class Metabolism(nn.Module):
    def __init__(self, state_dict=None):
        super(Metabolism, self).__init__()
        self.fc = None
        self.load_weights(state_dict)

    def load_weights(self, state_dict):
        if state_dict is None:
            self.fc = nn.Linear(1, 2)
        else:
            self.load_state_dict(state_dict)

    def forward(self, x):
        self.mutate_output()
        module = [m for m in self.modules()]
        print("=======FORWARD=============")
        print(module)
        for layer in module:
            if type(layer) is not Metabolism and not layer == module[-1]:
                x = F.selu(layer(x))
            elif layer == module[-1]:
                x = F.softmax(layer(x))
        return x

    def mutate_length(self):
        module = [m for m in self.modules() if not type(m) == Metabolism]
        self.add_module("fc"+str(len(module)), nn.Linear(module[-1].out_features, np.random.randint(2, len(module)+2)))

    def mutate_input(self):
        module = [m for m in self.modules() if not type(m) == Metabolism]

        # choose a layer to manipulate
        module_to_change = np.random.choice(module)
        module_tmp = copy.deepcopy(module_to_change)
        idx = module.index(module_to_change)
        # adding an input weight
        # module[idx].in_features += 1
        module[idx] = nn.Linear(module[idx].in_features+1, module[idx].out_features)

        # overwrite random weights with old weights
        for i in range(module_tmp.weight.size(0)):
            for j in range(module_tmp.weight.size(1)):
                with torch.no_grad():
                    module[idx].weight[i, j] = module_tmp.weight[i, j]
        module[idx].weight = torch.cat((module[idx].weight, torch.randn(module[idx].weight.size(1))), 1)

        # if its not the first layer, the fore layer needs to be adapted
        if idx > 0:
            module_tmp = copy.deepcopy(module[idx-1])

            # adding an input weight
            # module[idx-1].out_features += 1
            module[idx-1] = nn.Linear(module[idx-1].in_features, module[idx-1].out_features+1)

            # overwrite random weights with old weights
            for i in range(module_tmp.weight.size(0)):
                for j in range(module_tmp.weight.size(1)):
                    with torch.no_grad():
                        module[idx-1].weight[i, j] = module_tmp.weight[i, j]
            module[idx].weight = torch.cat((module[idx].weight, torch.randn(module[idx].weight.size(0))), 0)

    def mutate_output(self):
        module = [m for m in self.modules() if not type(m) == Metabolism]

        # choose a layer to manipulate
        module_to_change = np.random.choice(module)
        module_tmp = copy.deepcopy(module_to_change)
        idx = module.index(module_to_change)
        print("=============================")
        print(module[idx].weight)
        # adding an input weight
        # module[idx].out_features += 1
        module[idx] = nn.Linear(module[idx].in_features, module[idx].out_features + 1)
        # overwrite random weights with old weights
        print(idx)
        print(module_tmp.weight.size(0))
        print(module_tmp.weight.size(1))
        for i in range(module_tmp.weight.size(0)):
            for j in range(module_tmp.weight.size(1)):
                with torch.no_grad():
                    module[idx].weight[i, j] = module_tmp.weight[i, j]
        print(module[idx].weight)
        print(module[idx].weight.size(0))
        print(module[idx].weight.size(1))
        print("=============================")
        # if its not the first layer, the fore layer needs to be adapted
        if idx < len(module)-1:
            module_tmp = copy.deepcopy(module[idx + 1])
            # adding an input weight
            # module[idx+1].in_features += 1
            module[idx+1] = nn.Linear(
                module[idx+1].in_features+1,
                module[idx+1].out_features
                )

            # overwrite random weights with old weights
            for i in range(module_tmp.weight.size(0)):
                for j in range(module_tmp.weight.size(1)):
                    with torch.no_grad():
                        module[idx+1].weight[i, j] = module_tmp.weight[i, j]
        module = [m for m in self.modules() if not type(m) == Metabolism]
        print(module[idx].weight)
        return idx

    def mutate_weights(self):
        for module in self.modules():
            if type(module) == Metabolism:
                for i in range(module.fc.weight.size(0)):
                    for j in range(module.fc.weight.size(1)):
                        if np.random.randint(0, 10) == 0:
                            with torch.no_grad():
                                module.fc.weight[i, j] += np.random.normal(0, 0.001)

    def get_fathers_genes(self, father):
        for module in self.modules():
            if type(module) == Metabolism:
                for i in range(module.fc.weight.size(0)):
                    for j in range(module.fc.weight.size(1)):
                        if not np.random.randint(0, 2):
                            module.fc.weight[i, j] = father.fc.weight[i, j]
                            module.fc.bias[i] = father.fc.bias[i]


if __name__ == "__main__":
    model = Metabolism()
    rand_torch = torch.randn(1, 1)
    print("=================MOTHER=================")
    print(model.state_dict())
    # print(model.forward(rand_torch))

    print("=================FATHER=================")
    model2 = Metabolism()
    print(model2.state_dict())
    # print(model2.forward(rand_torch))

    print("=================CHILD=================")
    model_child = copy.deepcopy(model)
    model_child.get_fathers_genes(model2)
    print(model_child.state_dict())
    # print(model_child.forward(rand_torch))

    print("=================MUTATED=================")
    model_child.mutate_weights()
    print([m for m in model_child.modules() if not type(m) == Metabolism])
    # print(model_child.state_dict())
    # print(model_child.forward(rand_torch))

    model_child.mutate_length()
    model_child.mutate_length()
    print([m for m in model_child.modules() if not type(m) == Metabolism])
    print(model_child.state_dict())
    # print(model_child.forward(rand_torch))

    idx = model_child.mutate_output()
    module = [m for m in model_child.modules() if not type(m) == Metabolism]
    print(module[idx].weight)
    print(module)
    print(model_child.state_dict())
    print(model_child.forward(rand_torch))


