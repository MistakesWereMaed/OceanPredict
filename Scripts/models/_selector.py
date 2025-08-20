from . import PINN, GNN, FNO

def select_model(model_type):
    match model_type:
        case "PINN":
            model_class = PINN.model
        case "GNN":
            model_class = GNN.model
        case "FNO":
            model_class = FNO.model
        case _:
            raise ValueError(f"Unknown model type")

    return model_class

def initialize_model(model_type, image_size, config=None):
    model_class = select_model(model_type)
        
    if config:
        params = config
    else:
        params = model_class.load_params()

    params["image_size"] = image_size

    model = model_class(**params)
    batch_size = params["batch_size"]

    return model, batch_size