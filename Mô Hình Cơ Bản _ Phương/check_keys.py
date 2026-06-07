import torch
checkpoint = torch.load("best_model.pth", map_location="cpu")
if isinstance(checkpoint, dict):
    print("Keys in state_dict:")
    for key in list(checkpoint.keys())[:10]:
        print(key)
else:
    print(f"Type of checkpoint: {type(checkpoint)}")
