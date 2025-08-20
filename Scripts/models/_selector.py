from . import PINN, GNN, FNO

def initialize_model(model_type, image_size):
    match model_type:
        case "PINN":
            model_class = PINN.model
        case "GNN":
            model_class = GNN.model
        case "FNO":
            model_class = FNO.model
        case _:
            raise ValueError(f"Unknown model type")
        
    params = model_class.load_params()
    params["image_size"] = image_size

    model = model_class(**params)
    batch_size = params["batch_size"]

    return model, batch_size