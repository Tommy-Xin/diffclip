from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from transformers.utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from .build import load_sd_model, load_clip_model_OpenAICLIPFATR
from .utils import initiate_time_steps, prepare_class_text_embeddings
import torchvision.transforms as transforms
import clip


@dataclass
class SDOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None


class SDModelFATR(PreTrainedModel):
    """
    SD Model with OpenAI CLIP using Frequency-Aware Token Reduction (FATR).
    This is identical to SDModel but uses OpenAICLIPFATR instead of OpenAICLIP.
    """
    def __init__(
        self,
        config = None,
    ):
        super().__init__(config)
        
        self.model_id, self.pipe, self.vae, self.tokenizer, self.text_encoder, self.unet, self.scheduler, self.image_renormalizer = load_sd_model(config)
        self.text_encoder.eval()
        self.vae.eval()
        self.unet.eval()
        
        self.pattern_dictionary={'None':['']}
        self.config.actual_bs = len(self.pattern_dictionary[self.config.visual_pattern])
        # Use FATR version of CLIP model
        self.class_model = load_clip_model_OpenAICLIPFATR(config)
        self.class_model.eval()
        self.config = config
        discrimi_size = self.config.clip_image_size
        self.resize_transform_discrimi = transforms.Resize((discrimi_size, discrimi_size))
        self.visual_proj = nn.Linear(768, 1024)
        
        # Optional: Separate projections for different token types (uncomment to enable)
        # self.class_token_proj = nn.Linear(768, 1024)
        # self.hf_tokens_proj = nn.Linear(768, 1024)
        # self.dc_tokens_proj = nn.Linear(768, 1024)

    def classify(self, image, classes):

        image_features, logits = self.class_model(image)

        if classes is not None:
            logits = logits[:, classes]

        probs = logits.softmax(-1)
        max_idx = probs.argmax(-1)
        K = probs.shape[-1] if self.config.tta.adapt_topk == -1 else self.config.tta.adapt_topk
        topk_idx = probs.argsort(descending=True)[:, :K]

        if classes is not None:
            classes = torch.tensor(classes).to(logits.device)
            max_class_idx = classes[max_idx.flatten()].view(max_idx.shape)
            topk_class_idx = classes[topk_idx.flatten()].view(topk_idx.shape)
        else:
            max_class_idx, topk_class_idx = max_idx, topk_idx

        return image_features, logits, topk_idx, max_class_idx, topk_class_idx
    
    def extract_fatr_tokens(self, image_features, tau=0.25, w=1, target_num_tokens=None):
        """
        Extract and separate different token types from FATR image features.
        
        FATR token structure: [B, N_tokens, dim] where N_tokens = 1 + r + w^2
        Token order: [class_token, HF_tokens (r tokens), DC_tokens (w^2 tokens)]
        
        Args:
            image_features: [B, N_tokens, dim] - Output from FATR model
            tau: reduction ratio (used to calculate r if target_num_tokens is None)
            w: number of DC token groups (w^2 DC tokens)
            target_num_tokens: target total token count (including class token)
        
        Returns:
            class_token: [B, 1, dim] - Class token (global image representation)
            hf_tokens: [B, r, dim] - High-frequency tokens
            dc_tokens: [B, w^2, dim] - Direct Current tokens (aggregated from LF tokens)
        """
        if image_features.dim() != 3:
            raise ValueError(f"Expected 3D tensor [B, N_tokens, dim], got {image_features.dim()}D")
        
        B, N_tokens, dim = image_features.shape
        
        # Extract class token (first token)
        class_token = image_features[:, 0:1, :]  # [B, 1, dim]
        
        # Remaining tokens are patch tokens: [HF_tokens, DC_tokens]
        patch_tokens = image_features[:, 1:, :]  # [B, N_tokens-1, dim]
        num_patch_tokens = patch_tokens.shape[1]
        
        # Calculate r (number of HF tokens)
        if target_num_tokens is not None:
            # target_num_tokens = 1 (class) + r (HF) + w^2 (DC)
            r = target_num_tokens - 1 - w * w
        else:
            # Estimate r based on tau (approximate, since we don't know original patch count)
            # This is an approximation - actual r depends on original patch count
            r = max(1, int(num_patch_tokens * tau / (1 + tau)))
        
        r = max(1, min(r, num_patch_tokens - w * w))  # Ensure valid range
        
        # Extract HF tokens and DC tokens
        hf_tokens = patch_tokens[:, :r, :]  # [B, r, dim] - High-frequency tokens
        dc_tokens = patch_tokens[:, r:, :]  # [B, w^2, dim] - Direct Current tokens
        
        return class_token, hf_tokens, dc_tokens
    
    def _unet_pred_noise(self, x_start, t, noise, context):

        _,c,h,w = x_start.shape
        device = t.device
        nt = t.shape[0]

        x_start = x_start.unsqueeze(1)
        x_start = x_start.expand(-1, nt//x_start.shape[0], -1, -1, -1)
        x_start = x_start.reshape(-1,c,h,w)

        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        noised_latent = (
            x_start * (alphas_cumprod[t]**0.5).view(-1, 1, 1, 1).to(device)
            + noise * ((1 - alphas_cumprod[t])**0.5).view(-1, 1, 1, 1).to(device)
        )
        pred_noise = self.unet(noised_latent, t, encoder_hidden_states=context.expand(nt, -1, -1)).sample

        return pred_noise
    
    def zeroshot_classifier(self, classnames, templates, model):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames:
                texts = [template.format(classname) for template in templates]
                texts = clip.tokenize(texts, truncate=True).cuda()
                class_embeddings = model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    def forward(
        self,
        image: torch.Tensor = None,
        text = None
    ) -> SDOutput:
        
        text = self.pattern_dictionary[self.config.visual_pattern]
        with torch.no_grad():
            imagenet_templates = ['{}',]
            zeroshot_weights = self.zeroshot_classifier(text, imagenet_templates, self.class_model.model.float()).float()
            
        self.class_model.final_fc.weight.data = zeroshot_weights.T
        self.class_model.final_fc.weight.data = self.class_model.final_fc.weight.data.contiguous()
        classes = [i for i in range(len(text))]
        
        discrimi_image = self.resize_transform_discrimi(image)
        genera_image = image
        real_BS = image.shape[0]
        after_DF_expand_BS = real_BS*self.config.input.batch_size
        
        # prepare_vae_latent
        self.vae, self.text_encoder, self.unet = self.vae.to(torch.float32), self.text_encoder.to(torch.float32), self.unet.to(torch.float32)
        renormed_image = self.image_renormalizer(genera_image).detach()
        x0 = self.vae.encode(renormed_image).latent_dist.mean.float()
        latent = x0 * 0.18215
        
        # prepare_total_timesteps
        total_timestep = self.scheduler.num_train_timesteps
        
        for step in range(self.config.tta.gradient_descent.train_steps):
            # Initiate timesteps and noise
            timesteps = initiate_time_steps(step, total_timestep, after_DF_expand_BS, self.config).long()
            timesteps = timesteps.cuda()

            c, h, w = latent.shape[1:]
            if not self.config.tta.use_same_noise_among_timesteps:
                noise = torch.randn((real_BS* self.config.input.batch_size, c, h, w)).cuda()
            else:
                noise = torch.randn((1, c, h, w)).cuda()
                noise = noise.repeat(real_BS* self.config.input.batch_size, 1, 1, 1)
            
            if self.config.tta.adapt_topk == -1:
                image_features, logits, _, _, _ = self.classify(discrimi_image, classes)
                pred_top_idx = None
            else:
                image_features, logits, pred_top_idx, _, _ = self.classify(discrimi_image, classes)
            real_BS, C = logits.shape[:2]

            # Pick top-K predictions
            if pred_top_idx is not None:
                pred_top_idx = pred_top_idx.squeeze(0)
            else:
                pred_top_idx = torch.arange(C).cuda()

            logits = logits[:, pred_top_idx]

            class_text_embeddings = prepare_class_text_embeddings(self.tokenizer, self.text_encoder, class_names=text)
            class_text_embeddings = class_text_embeddings.detach()
            class_text_embeddings = class_text_embeddings[pred_top_idx, :]

            # Compute conditional text embeddings using weighted-summed predictions
            probs = logits.softmax(-1)
            probs = probs[:, :, None, None]
            class_text_embeddings = (class_text_embeddings.unsqueeze(0).repeat(after_DF_expand_BS, 1, 1, 1))
            _, word_num, _, _ = probs.shape
            probs = probs.unsqueeze(1).repeat(1,self.config.input.batch_size,1,1,1).reshape(-1,word_num,1,1)
            context = (probs * class_text_embeddings).sum(1)
            
            # FATR token structure: [B, N_tokens, dim] where N_tokens = 1 + r + w^2
            # Token order: [class_token, HF_tokens (r tokens), DC_tokens (w^2 tokens)]
            # - class_token: [B, 1, dim] - Global image representation (position 0)
            # - HF_tokens: [B, r, dim] - High-frequency tokens (positions 1 to r)
            # - DC_tokens: [B, w^2, dim] - Direct Current tokens aggregated from LF tokens (positions r+1 to r+w^2)
            
            # ====================================================================
            # Method 1: Separate token processing (RECOMMENDED - uncomment to use)
            # ====================================================================
            use_separate_token_processing = False  # Set to True to enable separate processing
            
            if use_separate_token_processing and image_features.dim() == 3:
                # Extract different token types using the helper method
                tau = getattr(self.class_model.model.visual, 'tau', 0.25)
                w = getattr(self.class_model.model.visual, 'w', 1)
                target_num = getattr(self.class_model.model.visual, 'target_num_tokens', None)
                class_token, hf_tokens, dc_tokens = self.extract_fatr_tokens(
                    image_features, tau=tau, w=w, target_num_tokens=target_num
                )
                
                # Project each token type separately
                class_token_proj = self.visual_proj(class_token)  # [B, 1, 1024]
                hf_tokens_proj = self.visual_proj(hf_tokens)  # [B, r, 1024]
                dc_tokens_proj = self.visual_proj(dc_tokens)  # [B, w^2, 1024]
                
                # Reconstruct image features: [class_token, HF_tokens, DC_tokens]
                image_features_proj = torch.cat([class_token_proj, hf_tokens_proj, dc_tokens_proj], dim=1)  # [B, 1+r+w^2, 1024]
                
                # Pad to match context length (77)
                B, N_tokens, dim = image_features_proj.shape
                target_length = context.shape[1]  # 77
                if N_tokens < target_length:
                    padding_size = target_length - N_tokens
                    # Strategy: Use DC token mean as padding (contains global low-frequency info)
                    padding_value = dc_tokens_proj.mean(dim=1, keepdim=True)  # [B, 1, 1024]
                    padding = padding_value.expand(B, padding_size, dim)  # [B, padding_size, 1024]
                    image_features_proj = torch.cat([image_features_proj, padding], dim=1)
                elif N_tokens > target_length:
                    image_features_proj = image_features_proj[:, :target_length, :]
                
                image_features = image_features_proj  # [B, 77, 1024]
            
            # ====================================================================
            # Method 2: Current implementation - treat all tokens together (DEFAULT)
            # ====================================================================
            elif image_features.dim() == 3:
                # image_features: [B, N_tokens, dim]
                B, N_tokens, dim = image_features.shape
                target_length = context.shape[1]  # Should be 77 (text context length)
                
                # Separate different token types for clarity
                class_token = image_features[:, 0:1, :]  # [B, 1, dim] - Class token
                patch_tokens = image_features[:, 1:, :]  # [B, N_tokens-1, dim] - Contains HF + DC tokens
                
                # Note: In FATR, patch_tokens = [HF_tokens, DC_tokens]
                # The exact split depends on tau and w parameters:
                # - HF tokens count: r = floor(N_patch * tau) or target_num - w^2
                # - DC tokens count: w^2 (where w is typically 1 for global DC)
                # For w=1: patch_tokens = [HF_tokens (r), DC_token (1)]
                # For w>1: patch_tokens = [HF_tokens (r), DC_tokens (w^2)]
                
                if N_tokens < target_length:
                    # Pad to match target length (77 for text context)
                    padding_size = target_length - N_tokens
                    # Use mean of all tokens as padding value (better than zeros)
                    padding_value = image_features.mean(dim=1, keepdim=True)  # [B, 1, dim]
                    padding = padding_value.expand(B, padding_size, dim)  # [B, padding_size, dim]
                    image_features = torch.cat([image_features, padding], dim=1)  # [B, 77, dim]
                elif N_tokens > target_length:
                    # Truncate if more than target length (shouldn't happen with FATR)
                    image_features = image_features[:, :target_length, :]
                # If N_tokens == target_length, no change needed
            elif image_features.dim() == 2:
                # If 2D [B, dim], expand to [B, 1, dim] then pad
                # This case shouldn't happen with FATR, but handle for compatibility
                image_features = image_features.unsqueeze(1)  # [B, 1, dim]
                B, _, dim = image_features.shape
                target_length = context.shape[1]
                padding_size = target_length - 1
                padding_value = image_features.expand(B, padding_size, dim)
                image_features = torch.cat([image_features, padding_value], dim=1)  # [B, 77, dim]
            
            image_features = self.visual_proj(image_features)  # [B, 77, 1024]
            context = context + image_features  # [B, 77, 1024]

            # Predict noise with the diffusion model
            pred_noise = self._unet_pred_noise(x_start=latent, t=timesteps, noise=noise, context=context).float()

            # Compute diffusion loss
            if self.config.tta.loss == "l1":
                loss = torch.nn.functional.l1_loss(pred_noise, noise)
            else:
                loss = torch.nn.functional.mse_loss(pred_noise, noise)
            
            if step != (self.config.tta.gradient_descent.train_steps-1):
                loss.backward()

        return SDOutput(loss=loss)

