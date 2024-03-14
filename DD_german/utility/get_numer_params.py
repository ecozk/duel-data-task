def count_parameters(model):
    "count number of trainable parameter for a model"
    return sum(p.numel() for p in model.parameters() if p.requires_grad)