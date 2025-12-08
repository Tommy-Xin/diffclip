import torch
state = torch.load("pretrained_weights/CLIP/OpenAICLIP/OpenAI-ViT-L-14-224.pth", map_location="cpu")
print(type(state))
print(list(state.keys())[:20])