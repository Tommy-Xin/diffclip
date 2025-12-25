import clip
import torch.nn as nn
from open_clip import create_model_from_pretrained, create_model_and_transforms
from clip.model import build_model
import torch
import sys
import os

# Add parent directory to path for importing model_fatr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from CLIP.clip.model_fatr import build_model_fatr
class OpenAICLIP(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.clip_image_size == 224:
            #model, _ = clip.load("pretrained_weights/CLIP/OpenAICLIP/OpenAI-ViT-L-14-224.pth", jit=False) #pretrained_weights/CLIP/ViT-L/14pretrained_weights/CLIP/ViT-L/14 #
            #model, _ = clip.load("/data/home/gaiyiming/hjq/xinc/DiffusionCLIP/pretrained_weights/CLIP/OpenAICLIP/OpenAI-ViT-L-14-224.pth" ,jit=False)
            state = torch.load('pretrained_weights/CLIP/OpenAICLIP/OpenAI-ViT-L-14-224.pth', map_location="cpu")
            model = build_model(state)
        if config.clip_image_size == 336:
            #model, _ = clip.load("pretrained_weights/CLIP/OpenAICLIP/ViT-L-14@336px.pt",jit=False)
            model, _ = clip.load("ViT-L/14@336px",jit=False)

        self.final_fc = nn.Linear(768, config.actual_bs, bias=False)
        self.model = model
        self.config = config

    # def forward(self, images):
        
    #     image_features = self.model.encode_image(images).float()
    #     logits = 100. * self.final_fc(image_features[:,0,:]).float()

    #     return image_features, logits
    def forward(self, images):
        # revise the dimension here
        image_features = self.model.encode_image(images).float()
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)
        pooled = image_features[:, 0, :]
        logits = 100. * self.final_fc(pooled).float()

        # logits = 100. * self.final_fc(image_features[:,0,:]).float()

        return image_features, logits


class OpenAICLIPFATR(nn.Module):
    """
    OpenAI CLIP with Frequency-Aware Token Reduction (FATR).
    Compatible with original OpenAICLIP interface, but uses FATR for token reduction.
    """
    def __init__(self, config):
        super().__init__()

        # FATR parameters (can be configured via config)
        # If None, auto-calculate based on tau. Set to 77 to align with text encoder
        target_num_tokens = getattr(config, 'fatr_target_num_tokens', None)
        tau = getattr(config, 'fatr_tau', 0.25)
        w = getattr(config, 'fatr_w', 1)

        if config.clip_image_size == 224:
            state = torch.load('pretrained_weights/CLIP/OpenAICLIP/OpenAI-ViT-L-14-224.pth', map_location="cpu")
            # Use build_model_fatr instead of build_model
            model = build_model_fatr(
                state, 
                target_num_tokens=target_num_tokens,
                tau=tau,
                w=w
            )
        elif config.clip_image_size == 336:
            # For 336px, we need to load the model first, then convert
            state = torch.load("pretrained_weights/CLIP/OpenAICLIP/ViT-L-14-336px.pt", map_location="cpu")
            # Note: For 336px, FATR might need special handling
            # For now, we'll use the original model and apply FATR manually if needed
            # You may need to extract state_dict and use build_model_fatr
            model = build_model_fatr(
                state,
                target_num_tokens=target_num_tokens,
                tau=tau,
                w=w
            )
        else:
            raise ValueError(f"Unsupported clip_image_size: {config.clip_image_size}")

        self.final_fc = nn.Linear(768, config.actual_bs, bias=False)
        self.model = model
        self.config = config

    def forward(self, images):
        # Same interface as OpenAICLIP
        image_features = self.model.encode_image(images).float()
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)
        pooled = image_features[:, 0, :]
        logits = 100. * self.final_fc(pooled).float()

        return image_features, logits


class DFN(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.clip_image_size == 224:
            model, _ = create_model_from_pretrained(model_name='ViT-H-14-quickgelu', pretrained="pretrained_weights/CLIP/DFN5B-CLIP-ViT-H-14/open_clip_pytorch_model.bin")

        if config.clip_image_size == 378:
            model, _ = create_model_from_pretrained(model_name='ViT-H-14-378-quickgelu', pretrained="pretrained_weights/CLIP/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin")
        
        self.final_fc = nn.Linear(1024, config.actual_bs, bias=False)
        self.model = model
        self.config = config

    def forward(self, images):
        
        image_features = self.model.encode_image(images).float()
        logits = 100. * self.final_fc(image_features[:,0,:]).float()    

        return image_features, logits
    
    
class SigLIP(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.clip_image_size == 224:
            model, _ = create_model_from_pretrained(model_name='ViT-SO400M-14-SigLIP', pretrained="pretrained_weights/CLIP/ViT-SO400M-14-SigLIP/open_clip_pytorch_model.bin",
                                                    image_mean=([0.5,0.5,0.5]), image_std=([0.5,0.5,0.5]), image_interpolation="bicubic", image_resize_mode="squash")
        if config.clip_image_size == 384:
            model, _ = create_model_from_pretrained(model_name='ViT-SO400M-14-SigLIP-384', pretrained="pretrained_weights/CLIP/ViT-SO400M-14-SigLIP-384/open_clip_pytorch_model.bin",
                                                     image_mean=([0.5,0.5,0.5]), image_std=([0.5,0.5,0.5]), image_interpolation="bicubic", image_resize_mode="squash")

        self.final_fc = nn.Linear(1152, config.actual_bs, bias=False)
        self.model = model
        self.config = config

    def forward(self, images):
        
        image_features = self.model.encode_image(images).float()
        logits = 100. * self.final_fc(image_features[:,0,:]).float()    

        return image_features, logits


class MetaCLIP(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.metaclip_version == "large":
            model, _, _ = create_model_and_transforms(model_name='ViT-L-14-quickgelu', pretrained="pretrained_weights/CLIP/MetaCLIP/l14_fullcc2.5b.pt")
            self.final_fc = nn.Linear(768, config.actual_bs, bias=False)
        if config.metaclip_version == "huge":
            model, _, _ = create_model_and_transforms(model_name='ViT-H-14-quickgelu', pretrained="pretrained_weights/CLIP/MetaCLIP/h14_fullcc2.5b.pt")
            self.final_fc = nn.Linear(1024, config.actual_bs, bias=False)

        self.model = model
        self.config = config

    def forward(self, images):
        
        image_features = self.model.encode_image(images).float()
        logits = 100. * self.final_fc(image_features[:,0,:]).float()    

        return image_features, logits
