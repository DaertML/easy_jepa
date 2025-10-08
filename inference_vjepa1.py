import os
import sys
import argparse
import yaml
import time
import copy
import tempfile
import csv
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from src.models.vision_transformer import VisionTransformer
from src.masks.random_tube import MaskCollator as TubeMaskCollator
from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
from src.masks.utils import apply_masks
from app.vjepa.transforms import make_transforms
from app.vjepa.utils import init_video_model
from src.datasets.data_manager import init_data

from src.utils.logging import get_logger
from src.utils.tensors import repeat_interleave_batch
import torchvision.transforms.functional as TF

import os
import sys
import argparse
import yaml
import time
import copy
import tempfile
import csv
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.models.vision_transformer import VisionTransformer
from src.masks.random_tube import MaskCollator as TubeMaskCollator
from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
from src.masks.utils import apply_masks
from app.vjepa.transforms import make_transforms
from app.vjepa.utils import init_video_model
from src.datasets.data_manager import init_data

from src.utils.logging import get_logger
from src.utils.tensors import repeat_interleave_batch
import torchvision.transforms.functional as TF


logger = get_logger(__name__)

class AttentiveProbe(nn.Module):
    """Attentive probe for action classification on Something-Something dataset"""
    
    def __init__(self, embed_dim=1024, num_classes=174, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        
        # Multi-head attention for temporal aggregation
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Learnable classification token (like in ViT)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
    def forward(self, x):
        """
        x: [B, T, D] where T is temporal dimension (frames or patches), D is feature dimension
        """
        B, T, D = x.shape
        
        # Add classification token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, D]
        
        # Apply temporal attention
        attn_output, attn_weights = self.temporal_attention(
            query=x, key=x, value=x
        )
        
        # Use the classification token for prediction
        cls_output = attn_output[:, 0]  # [B, D]
        
        # Classification
        logits = self.classifier(cls_output)  # [B, num_classes]
        
        return logits, attn_weights

# Load Something-Something v2 label mapping
def load_ssv2_labels():
    """Load Something-Something v2 action labels"""
    # This is a subset of common SSv2 labels - you may need to adjust based on your specific labels
    """ssv2_labels = {
        0: "Putting something on a surface",
        1: "Pushing something from left to right",
        2: "Pushing something from right to left", 
        3: "Moving something up",
        4: "Moving something down",
        5: "Pulling something from left to right",
        6: "Pulling something from right to left",
        7: "Lifting something up",
        8: "Dropping something",
        9: "Poking something so that it falls over",
        10: "Poking something so that it slightly moves",
        11: "Pushing something with something",
        12: "Moving something towards something",
        13: "Moving something away from something",
        14: "Pretending to put something on a surface",
        15: "Pretending to take something from a surface",
        # Add more labels as needed for your specific probe
    }"""

    ssv2_labels = {
   0:"Approaching [something] with your camera",
   1:"Attaching [something] to [something]",
   2:"Bending [something] so that it deforms",
   3:"Bending [something] until it breaks",
   4:"Burying [something] in [something]",
   5:"Closing [something]",
   6:"Covering [something] with [something]",
   7:"Digging [something] out of [something]",
   8:"Dropping [something] behind [something]",
   9:"Dropping [something] in front of [something]",
   10:"Dropping [something] into [something]",
   11:"Dropping [something] next to [something]",
   12:"Dropping [something] onto [something]",
   13:"Failing to put [something] into [something] because [something] does not fit",
   14:"Folding [something]",
   15:"Hitting [something] with [something]",
   16:"Holding [something]",
   17:"Holding [something] behind [something]",
   18:"Holding [something] in front of [something]",
   19:"Holding [something] next to [something]",
   20:"Holding [something] over [something]",
   21:"Laying [something] on the table on its side, not upright",
   22:"Letting [something] roll along a flat surface",
   23:"Letting [something] roll down a slanted surface",
   24:"Letting [something] roll up a slanted surface, so it rolls back down",
   25:"Lifting a surface with [something] on it but not enough for it to slide down",
   26:"Lifting a surface with [something] on it until it starts sliding down",
   27:"Lifting [something] up completely without letting it drop down",
   28:"Lifting [something] up completely, then letting it drop down",
   29:"Lifting [something] with [something] on it",
   30:"Lifting up one end of [something] without letting it drop down",
   31:"Lifting up one end of [something], then letting it drop down",
   32:"Moving away from [something] with your camera",
   33:"Moving [part] of [something]",
   34:"Moving [something] across a surface until it falls down",
   35:"Moving [something] across a surface without it falling down",
   36:"Moving [something] and [something] away from each other",
   37:"Moving [something] and [something] closer to each other",
   38:"Moving [something] and [something] so they collide with each other",
   39:"Moving [something] and [something] so they pass each other",
   40:"Moving [something] away from [something]",
   41:"Moving [something] away from the camera",
   42:"Moving [something] closer to [something]",
   43:"Moving [something] down",
   44:"Moving [something] towards the camera",
   45:"Moving [something] up",
   46:"Opening [something]",
   47:"Picking [something] up",
   48:"Piling [something] up",
   49:"Plugging [something] into [something]",
   50:"Plugging [something] into [something] but pulling it right out as you remove your hand",
   51:"Poking a hole into [some substance]",
   52:"Poking a hole into [something soft]",
   53:"Poking a stack of [something] so the stack collapses",
   54:"Poking a stack of [something] without the stack collapsing",
   55:"Poking [something] so it slightly moves",
   56:"Poking [something] so lightly that it doesn't or almost doesn't move",
   57:"Poking [something] so that it falls over",
   58:"Poking [something] so that it spins around",
   59:"Pouring [something] into [something]",
   60:"Pouring [something] into [something] until it overflows",
   61:"Pouring [something] onto [something]",
   62:"Pouring [something] out of [something]",
   63:"Pretending or failing to wipe [something] off of [something]",
   64:"Pretending or trying and failing to twist [something]",
   65:"Pretending to be tearing [something that is not tearable]",
   66:"Pretending to close [something] without actually closing it",
   67:"Pretending to open [something] without actually opening it",
   68:"Pretending to pick [something] up",
   69:"Pretending to poke [something]",
   70:"Pretending to pour [something] out of [something], but [something] is empty",
   71:"Pretending to put [something] behind [something]",
   72:"Pretending to put [something] into [something]",
   73:"Pretending to put [something] next to [something]",
   74:"Pretending to put [something] on a surface",
   75:"Pretending to put [something] onto [something]",
   76:"Pretending to put [something] underneath [something]",
   77:"Pretending to scoop [something] up with [something]",
   78:"Pretending to spread air onto [something]",
   79:"Pretending to sprinkle air onto [something]",
   80:"Pretending to squeeze [something]",
   81:"Pretending to take [something] from [somewhere]",
   82:"Pretending to take [something] out of [something]",
   83:"Pretending to throw [something]",
   84:"Pretending to turn [something] upside down",
   85:"Pulling [something] from behind of [something]",
   86:"Pulling [something] from left to right",
   87:"Pulling [something] from right to left",
   88:"Pulling [something] onto [something]",
   89:"Pulling [something] out of [something]",
   90:"Pulling two ends of [something] but nothing happens",
   91:"Pulling two ends of [something] so that it gets stretched",
   92:"Pulling two ends of [something] so that it separates into two pieces",
   93:"Pushing [something] from left to right",
   94:"Pushing [something] from right to left",
   95:"Pushing [something] off of [something]",
   96:"Pushing [something] onto [something]",
   97:"Pushing [something] so it spins",
   98:"Pushing [something] so that it almost falls off but doesn't",
   99:"Pushing [something] so that it falls off the table",
   100:"Pushing [something] so that it slightly moves",
   101:"Pushing [something] with [something]",
   102:"Putting [number of] [something] onto [something]",
   103:"Putting [something] and [something] on the table",
   104:"Putting [something] behind [something]",
   105:"Putting [something] in front of [something]",
   106:"Putting [something] into [something]",
   107:"Putting [something] next to [something]",
   108:"Putting [something] on a flat surface without letting it roll",
   109:"Putting [something] on a surface",
   110:"Putting [something] on the edge of [something] so it is not supported and falls down",
   111:"Putting [something] onto a slanted surface but it doesn't glide down",
   112:"Putting [something] onto [something]",
   113:"Putting [something] onto [something else that cannot support it] so it falls down",
   114:"Putting [something similar to other things that are already on the table]",
   115:"Putting [something] that can't roll onto a slanted surface, so it slides down",
   116:"Putting [something] that can't roll onto a slanted surface, so it stays where it is",
   117:"Putting [something that cannot actually stand upright] upright on the table, so it falls on its side",
   118:"Putting [something] underneath [something]",
   119:"Putting [something] upright on the table",
   120:"Putting [something], [something] and [something] on the table",
   121:"Removing [something], revealing [something] behind",
   122:"Rolling [something] on a flat surface",
   123:"Scooping [something] up with [something]",
   124:"Showing a photo of [something] to the camera",
   125:"Showing [something] behind [something]",
   126:"Showing [something] next to [something]",
   127:"Showing [something] on top of [something]",
   128:"Showing [something] to the camera",
   129:"Showing that [something] is empty",
   130:"Showing that [something] is inside [something]",
   131:"[Something] being deflected from [something]",
   132:"[Something] colliding with [something] and both are being deflected",
   133:"[Something] colliding with [something] and both come to a halt",
   134:"[Something] falling like a feather or paper",
   135:"[Something] falling like a rock",
   136:"Spilling [something] behind [something]",
   137:"Spilling [something] next to [something]",
   138:"Spilling [something] onto [something]",
   139:"Spinning [something] so it continues spinning",
   140:"Spinning [something] that quickly stops spinning",
   141:"Spreading [something] onto [something]",
   142:"Sprinkling [something] onto [something]",
   143:"Squeezing [something]",
   144:"Stacking [number of] [something]",
   145:"Stuffing [something] into [something]",
   146:"Taking [one of many similar things on the table]",
   147:"Taking [something] from [somewhere]",
   148:"Taking [something] out of [something]",
   149:"Tearing [something] into two pieces",
   150:"Tearing [something] just a little bit",
   151:"Throwing [something]",
   152:"Throwing [something] against [something]",
   153:"Throwing [something] in the air and catching it",
   154:"Throwing [something] in the air and letting it fall",
   155:"Throwing [something] onto a surface",
   156:"Tilting [something] with [something] on it slightly so it doesn't fall down",
   157:"Tilting [something] with [something] on it until it falls off",
   158:"Tipping [something] over",
   159:"Tipping [something] with [something in it] over, so [something in it] falls out",
   160:"Touching (without moving) [part] of [something]",
   161:"Trying but failing to attach [something] to [something] because it doesn't stick",
   162:"Trying to bend [something unbendable] so nothing happens",
   163:"Trying to pour [something] into [something], but missing so it spills next to it",
   164:"Turning [something] upside down",
   165:"Turning the camera downwards while filming [something]",
   166:"Turning the camera left while filming [something]",
   167:"Turning the camera right while filming [something]",
   168:"Turning the camera upwards while filming [something]",
   169:"Twisting (wringing) [something] wet until water comes out",
   170:"Twisting [something]",
   171:"Uncovering [something]",
   172:"Unfolding [something]",
   173:"Wiping [something] off of [something]"
}
    return ssv2_labels

def safe_to_numpy(tensor):
    """Safely convert tensor to numpy, handling BFloat16 and other types"""
    if tensor is None:
        return None
    tensor = tensor.detach().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    return tensor.numpy()

# [Keep the PixelDecoder class from previous implementation...]
class PixelDecoder(nn.Module):
    """Simple pixel decoder to reconstruct frames from latent predictions"""
    
    def __init__(self, embed_dim=1024, decoder_embed_dim=512, patch_size=16, num_frames=16, crop_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.decoder_embed_dim = decoder_embed_dim  # Store the dimension
        
        # Calculate spatial and temporal dimensions
        self.spatial_patches = crop_size // patch_size
        self.temporal_patches = num_frames // 2  # Assuming tubelet_size=2
        
        # Decoder layers
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        
        # Transformer decoder blocks
        decoder_depth = 4
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=decoder_embed_dim,
                nhead=8,
                dim_feedforward=decoder_embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ) for _ in range(decoder_depth)
        ])
        
        # Output projection to pixel values
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * patch_size * 3 * 2)  # 2 for tubelet_size
        
    def forward(self, z, masks_enc, masks_pred):
        """
        z: list of latent predictions [B, N_pred, D]
        masks_enc: list of encoder masks
        masks_pred: list of predictor masks
        """
        
        # Use the first latent prediction
        if isinstance(z, list):
            x = z[0]  # [B, N_pred, D]
        else:
            x = z
            
        B, N_pred, D = x.shape
        
        # Project to decoder dimension
        x = self.decoder_embed(x)  # [B, N_pred, decoder_embed_dim]
        
        # Create positional embeddings for all patches
        num_patches = self.spatial_patches * self.spatial_patches * self.temporal_patches
        pos_embed = self.create_positional_embedding(B, num_patches, x.device)  # [B, num_patches, decoder_embed_dim]
        
        # Apply transformer decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x, pos_embed)
        
        x = self.decoder_norm(x)
        
        # Predict pixel values for each patch
        x = self.decoder_pred(x)  # [B, N_pred, patch_size*patch_size*3*2]
        
        # Reconstruct the full frame by placing predicted patches
        reconstructed = self.reconstruct_frames(x, masks_pred[0] if masks_pred else None)
        
        return reconstructed
    
    def create_positional_embedding(self, batch_size, num_patches, device):
        """Create simple positional embeddings"""
        # FIX: Use self.decoder_embed_dim instead of self.decoder_embed.embed_dim
        pos_embed = torch.zeros(num_patches, self.decoder_embed_dim, device=device)
        position = torch.arange(0, num_patches, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.decoder_embed_dim, 2, device=device).float() * 
                           (-np.log(10000.0) / self.decoder_embed_dim))
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        pos_embed = pos_embed.unsqueeze(0).expand(batch_size, -1, -1)
        return pos_embed
    
    def reconstruct_frames(self, patch_predictions, mask_pred):
        """
        Reconstruct full frames from patch predictions
        patch_predictions: [B, N_pred, patch_size*patch_size*3*2]
        mask_pred: [B, N_pred] indices of predicted patches
        """
        B, N_pred, D = patch_predictions.shape
        patch_pixels = self.patch_size * self.patch_size * 3 * 2
        
        # Reshape to individual patches
        patches = patch_predictions.view(B * N_pred, -1)  # [B*N_pred, patch_pixels]
        patches = patches.view(B * N_pred, 2, 3, self.patch_size, self.patch_size)  # [B*N_pred, tubelet, C, H, W]
        
        # Create empty frames
        reconstructed_frames = torch.zeros(B, self.num_frames, 3, self.crop_size, self.crop_size, 
                                         device=patch_predictions.device)
        
        if mask_pred is not None:
            # Place predicted patches in their correct positions
            for b in range(B):
                pred_indices = mask_pred[b]  # [N_pred]
                
                for i, patch_idx in enumerate(pred_indices):
                    if i >= N_pred:
                        break
                        
                    # Convert flat index to spatial-temporal coordinates
                    patch_idx = patch_idx.item()
                    t_idx = patch_idx // (self.spatial_patches * self.spatial_patches)
                    s_idx = patch_idx % (self.spatial_patches * self.spatial_patches)
                    h_idx = s_idx // self.spatial_patches
                    w_idx = s_idx % self.spatial_patches
                    
                    # Calculate pixel coordinates
                    h_start = h_idx * self.patch_size
                    w_start = w_idx * self.patch_size
                    
                    # Place the tubelet (2 frames) in the reconstructed frames
                    for t_offset in range(2):
                        frame_idx = t_idx * 2 + t_offset
                        if frame_idx < self.num_frames:
                            reconstructed_frames[b, frame_idx, :, h_start:h_start+self.patch_size, 
                                               w_start:w_start+self.patch_size] = patches[b*N_pred + i, t_offset]
        
        return reconstructed_frames


def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Reverses normalization for viewing."""
    tensor = tensor.detach().cpu()
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Input tensor must have 3 dimensions (C, H, W), got {tensor.ndim}")

    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return TF.to_pil_image(tensor)


def plot_representations(z_context, h, z, masks_enc, masks_pred, output_path=None):
    """Plot various representations from the model"""
    
    # Convert to numpy for visualization - handle lists and BFloat16
    if isinstance(z_context, list):
        # Use the first context features for visualization
        z_context_np = safe_to_numpy(z_context[0]) if z_context else np.array([])
    else:
        z_context_np = safe_to_numpy(z_context) if z_context is not None else np.array([])
    
    h_np = [safe_to_numpy(h_i) for h_i in h] if h else []
    z_np = [safe_to_numpy(z_i) for z_i in z] if z else []
    
    # [Rest of the plot_representations function remains the same...]
    # Create a comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('VJEPA Model Representations Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Context features heatmap (first mask)
    if len(z_context_np) > 0 and z_context_np.shape[1] > 0:
        # Take first batch and first 50 features for visualization
        context_vis = z_context_np[0, :, :50]  # [N_ctx, 50]
        im1 = axes[0, 0].imshow(context_vis.T, aspect='auto', cmap='viridis')
        axes[0, 0].set_title('Context Features (First 50 dims)')
        axes[0, 0].set_xlabel('Patch Index')
        axes[0, 0].set_ylabel('Feature Dimension')
        plt.colorbar(im1, ax=axes[0, 0])
    else:
        axes[0, 0].text(0.5, 0.5, 'No context features', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Context Features')
    
    # Plot 2: Target vs Predicted similarity matrix (first mask)
    if len(h_np) > 0 and len(z_np) > 0:
        h_flat = h_np[0][0]  # [N_pred, D]
        z_flat = z_np[0][0]  # [N_pred, D]
        
        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(h_flat, z_flat)
        
        im2 = axes[0, 1].imshow(similarity, cmap='RdYlBu', vmin=-1, vmax=1)
        axes[0, 1].set_title('Target vs Predicted Similarity')
        axes[0, 1].set_xlabel('Predicted Patches')
        axes[0, 1].set_ylabel('Target Patches')
        plt.colorbar(im2, ax=axes[0, 1])
    else:
        axes[0, 1].text(0.5, 0.5, 'No target/predicted features', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Target vs Predicted Similarity')
    
    # Plot 3: Feature distribution statistics
    all_features = []
    feature_labels = []
    
    if len(z_context_np) > 0 and z_context_np.shape[1] > 0:
        context_flat = z_context_np[0].flatten()
        all_features.extend(context_flat[:1000])  # Sample for visualization
        feature_labels.extend(['Context'] * min(1000, len(context_flat)))
    
    if len(h_np) > 0:
        h_flat = h_np[0][0].flatten()
        all_features.extend(h_flat[:1000])
        feature_labels.extend(['Target'] * min(1000, len(h_flat)))
    
    if len(z_np) > 0:
        z_flat = z_np[0][0].flatten()
        all_features.extend(z_flat[:1000])
        feature_labels.extend(['Predicted'] * min(1000, len(z_flat)))
    
    if all_features:
        feature_data = np.array(all_features)
        feature_labels = np.array(feature_labels)
        
        # Create violin plot
        sns.violinplot(x=feature_labels, y=feature_data, ax=axes[0, 2])
        axes[0, 2].set_title('Feature Value Distribution')
        axes[0, 2].set_ylabel('Feature Values')
    else:
        axes[0, 2].text(0.5, 0.5, 'No features for distribution', 
                       ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Feature Value Distribution')
    
    # Plot 4: t-SNE visualization of patches (first mask)
    if len(h_np) > 0 and len(z_np) > 0:
        try:
            # Combine context, target and predicted features
            combined_features = []
            combined_labels = []
            
            if len(z_context_np) > 0 and z_context_np.shape[1] > 0:
                combined_features.append(z_context_np[0])
                combined_labels.extend(['Context'] * z_context_np[0].shape[0])
            
            combined_features.append(h_np[0][0])
            combined_labels.extend(['Target'] * h_np[0][0].shape[0])
            
            combined_features.append(z_np[0][0])
            combined_labels.extend(['Predicted'] * z_np[0][0].shape[0])
            
            combined_features = np.vstack(combined_features)
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, combined_features.shape[0]-1))
            tsne_results = tsne.fit_transform(combined_features)
            
            # Plot
            colors = {'Context': 'blue', 'Target': 'green', 'Predicted': 'red'}
            for label in colors.keys():
                mask = np.array(combined_labels) == label
                if mask.any():  # Only plot if there are points for this label
                    axes[1, 0].scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                                     c=colors[label], label=label, alpha=0.6)
            
            axes[1, 0].set_title('t-SNE Visualization of Features')
            axes[1, 0].set_xlabel('t-SNE 1')
            axes[1, 0].set_ylabel('t-SNE 2')
            axes[1, 0].legend()
            
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, f't-SNE failed: {str(e)}', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('t-SNE Visualization (Failed)')
    else:
        axes[1, 0].text(0.5, 0.5, 'Insufficient data for t-SNE', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('t-SNE Visualization')
    
    # Plot 5: Mask statistics
    if masks_enc and masks_pred:
        enc_mask_sizes = [m.shape[1] for m in masks_enc]
        pred_mask_sizes = [m.shape[1] for m in masks_pred]
        
        x_pos = np.arange(len(enc_mask_sizes))
        axes[1, 1].bar(x_pos - 0.2, enc_mask_sizes, 0.4, label='Encoder Masks', alpha=0.7)
        axes[1, 1].bar(x_pos + 0.2, pred_mask_sizes, 0.4, label='Predictor Masks', alpha=0.7)
        axes[1, 1].set_title('Mask Sizes')
        axes[1, 1].set_xlabel('Mask Index')
        axes[1, 1].set_ylabel('Number of Patches')
        axes[1, 1].legend()
        axes[1, 1].set_xticks(x_pos)
    else:
        axes[1, 1].text(0.5, 0.5, 'No mask data', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Mask Sizes')
    
    # Plot 6: Feature norms
    norms_data = []
    norms_labels = []
    
    if len(z_context_np) > 0 and z_context_np.shape[1] > 0:
        context_norms = np.linalg.norm(z_context_np[0], axis=1)
        norms_data.append(context_norms)
        norms_labels.append('Context')
    
    if len(h_np) > 0:
        target_norms = np.linalg.norm(h_np[0][0], axis=1)
        norms_data.append(target_norms)
        norms_labels.append('Target')
    
    if len(z_np) > 0:
        predicted_norms = np.linalg.norm(z_np[0][0], axis=1)
        norms_data.append(predicted_norms)
        norms_labels.append('Predicted')
    
    if norms_data:
        axes[1, 2].boxplot(norms_data, labels=norms_labels)
        axes[1, 2].set_title('Feature Norm Distribution')
        axes[1, 2].set_ylabel('L2 Norm')
    else:
        axes[1, 2].text(0.5, 0.5, 'No data for norms', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Feature Norm Distribution')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Representation analysis saved to {output_path}")
    else:
        plt.show()
    
    return fig

def print_model_results(video_tensor, z_context, h, z, masks_enc, masks_pred, video_path,probe_results=None):
    """Print detailed model results in terminal"""
    
    print("\n" + "="*80)
    print("VJEPA MODEL INFERENCE RESULTS")
    print("="*80)
    
    # Basic information
    print(f"\n📊 BASIC INFORMATION:")
    print(f"   Video: {os.path.basename(video_path)}")
    print(f"   Input shape: {video_tensor.shape}")
    print(f"   Number of encoder masks: {len(masks_enc)}")
    print(f"   Number of predictor masks: {len(masks_pred)}")
    print(f"   Number of latent predictions: {len(z)}")
    
    # Feature statistics
    print(f"\n📈 FEATURE STATISTICS:")
    
    # Context features - handle list case
    if z_context is not None:
        if isinstance(z_context, list):
            for i, z_ctx in enumerate(z_context):
                z_ctx_np = safe_to_numpy(z_ctx)
                print(f"   Context features (mask {i}):")
                print(f"     Shape: {z_ctx.shape}")
                print(f"     Mean: {z_ctx_np.mean():.4f} ± {z_ctx_np.std():.4f}")
                print(f"     Min/Max: {z_ctx_np.min():.4f} / {z_ctx_np.max():.4f}")
        else:
            z_ctx_np = safe_to_numpy(z_context)
            print(f"   Context features:")
            print(f"     Shape: {z_context.shape}")
            print(f"     Mean: {z_ctx_np.mean():.4f} ± {z_ctx_np.std():.4f}")
            print(f"     Min/Max: {z_ctx_np.min():.4f} / {z_ctx_np.max():.4f}")
    
    # Target features
    if h and len(h) > 0:
        for i, h_i in enumerate(h):
            h_np = safe_to_numpy(h_i)
            print(f"   Target features (mask {i}):")
            print(f"     Shape: {h_i.shape}")
            print(f"     Mean: {h_np.mean():.4f} ± {h_np.std():.4f}")
            print(f"     Min/Max: {h_np.min():.4f} / {h_np.max():.4f}")
    
    # Predicted features
    if z and len(z) > 0:
        for i, z_i in enumerate(z):
            z_np = safe_to_numpy(z_i)
            print(f"   Predicted features (mask {i}):")
            print(f"     Shape: {z_i.shape}")
            print(f"     Mean: {z_np.mean():.4f} ± {z_np.std():.4f}")
            print(f"     Min/Max: {z_np.min():.4f} / {z_np.max():.4f}")
    
    # Mask information
    print(f"\n🎭 MASK INFORMATION:")
    for i, (mask_enc, mask_pred) in enumerate(zip(masks_enc, masks_pred)):
        print(f"   Mask pair {i}:")
        print(f"     Encoder: {mask_enc.shape[1]} patches kept")
        print(f"     Predictor: {mask_pred.shape[1]} patches predicted")
        
        # Calculate masking ratio
        if z_context is not None:
            if isinstance(z_context, list) and i < len(z_context):
                total_context_patches = z_context[i].shape[1]
            elif not isinstance(z_context, list):
                total_context_patches = z_context.shape[1]
            else:
                total_context_patches = 0
            
            total_patches = total_context_patches + mask_enc.shape[1]
            if total_patches > 0:
                mask_ratio = mask_enc.shape[1] / total_patches
                print(f"     Masking ratio: {mask_ratio:.2%}")
    
    # Similarity analysis
    print(f"\n🔍 SIMILARITY ANALYSIS:")
    if h and z and len(h) > 0 and len(z) > 0:
        from sklearn.metrics.pairwise import cosine_similarity
        
        for i, (h_i, z_i) in enumerate(zip(h, z)):
            h_np = safe_to_numpy(h_i[0])  # [N_pred, D]
            z_np = safe_to_numpy(z_i[0])  # [N_pred, D]
            
            # Compute cosine similarities
            similarities = []
            for j in range(min(h_np.shape[0], z_np.shape[0])):
                sim = cosine_similarity([h_np[j]], [z_np[j]])[0][0]
                similarities.append(sim)
            
            similarities = np.array(similarities)
            print(f"   Mask {i} similarity:")
            print(f"     Mean cosine similarity: {similarities.mean():.4f}")
            print(f"     Std cosine similarity: {similarities.std():.4f}")
            print(f"     Min/Max similarity: {similarities.min():.4f} / {similarities.max():.4f}")
    
    if probe_results:
        print(f"\n🎯 ATTENTIVE PROBE RESULTS:")
        for i, result in enumerate(probe_results):
            print(f"   Probe {i}:")
            print(f"     Predicted action: {result['predicted_label']}")
            print(f"     Confidence: {result['confidence']:.4f}")
            print(f"     Top 5 predictions:")
            for j, (label, conf) in enumerate(result['top_predictions']):
                print(f"       {j+1}. {label}: {conf:.4f}")

    # Memory usage
    print(f"\n💾 MEMORY USAGE:")
    if torch.cuda.is_available():
        print(f"   GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    print("="*80 + "\n")


def load_attentive_probe(probe_path, embed_dim=1024, num_classes=174, device='cuda'):
    """Load the attentive probe model"""
    logger.info(f"Loading attentive probe from: {probe_path}")
    
    # Initialize probe model
    probe = AttentiveProbe(
        embed_dim=embed_dim,
        num_classes=num_classes,
        num_heads=8,
        dropout=0.1
    )
    
    # Load checkpoint
    if os.path.exists(probe_path):
        try:
            checkpoint = torch.load(probe_path, map_location='cpu')
            logger.info(f"Probe checkpoint keys: {list(checkpoint.keys())}")
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                probe_state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                probe_state_dict = checkpoint['state_dict']
            else:
                probe_state_dict = checkpoint
            
            # Clean state dict keys
            probe_state_dict_cleaned = {}
            for k, v in probe_state_dict.items():
                # Remove 'module.' prefix if present
                k_clean = k.replace('module.', '', 1)
                probe_state_dict_cleaned[k_clean] = v
            
            # Load state dict
            missing_keys, unexpected_keys = probe.load_state_dict(probe_state_dict_cleaned, strict=False)
            logger.info(f"Probe missing keys: {len(missing_keys)}")
            logger.info(f"Probe unexpected keys: {len(unexpected_keys)}")
            
            if missing_keys:
                logger.warning(f"Probe missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Probe unexpected keys: {unexpected_keys}")
            
        except Exception as e:
            logger.error(f"Error loading probe checkpoint: {e}")
            return None
    else:
        logger.error(f"Probe checkpoint not found: {probe_path}")
        return None
    
    probe.to(device).eval()
    logger.info("Attentive probe loaded successfully")
    return probe

def run_attentive_probe(probe, features, ssv2_labels, top_k=5):
    """Run attentive probe on features and return results"""
    results = []
    
    with torch.no_grad():
        for i, feature in enumerate(features):
            # Ensure feature has the right shape [B, T, D]
            if feature.dim() == 3:
                # Feature is already [B, T, D]
                pass
            elif feature.dim() == 2:
                # Feature is [B, D], add temporal dimension
                feature = feature.unsqueeze(1)  # [B, 1, D]
            else:
                logger.warning(f"Unexpected feature shape: {feature.shape}")
                continue
            
            # Run probe
            logits, attn_weights = probe(feature)
            probabilities = F.softmax(logits, dim=-1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, top_k, dim=-1)
            
            # Convert to lists
            top_probs = top_probs[0].cpu().numpy()
            top_indices = top_indices[0].cpu().numpy()
            
            # Get labels
            top_predictions = []
            for idx, prob in zip(top_indices, top_probs):
                label = ssv2_labels.get(idx, f"Unknown action {idx}")
                top_predictions.append((label, prob))
            
            # Store results
            result = {
                'feature_index': i,
                'predicted_label': top_predictions[0][0],
                'confidence': top_predictions[0][1],
                'top_predictions': top_predictions,
                'logits': logits.cpu(),
                'attention_weights': attn_weights.cpu()
            }
            results.append(result)
    
    return results

def check_video_file(video_path):
    """Check if video file exists and is readable"""
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False
    
    # Try to get video info using ffprobe or similar
    try:
        import subprocess
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,nb_frames,duration,r_frame_rate',
            '-of', 'csv=p=0', video_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            info = result.stdout.strip().split(',')
            logger.info(f"Video info: width={info[0]}, height={info[1]}, frames={info[2]}, duration={info[3]}s, fps={info[4]}")
        else:
            logger.warning(f"Could not get video info: {result.stderr}")
    except Exception as e:
        logger.warning(f"Could not check video info: {e}")
    
    return True

def main_inference(args):
    # --- 1. Load Configuration ---
    logger.info(f"Loading configuration from: {args.config_path}")
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    cfgs_meta = config.get('meta', {})
    cfgs_mask = config.get('mask', {})
    cfgs_model = config.get('model', {})
    cfgs_data = config.get('data', {})
    cfgs_data_aug = config.get('data_aug', {})

    which_dtype = cfgs_meta.get('dtype', 'float32')
    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}.get(which_dtype.lower(), torch.float32)
    logger.info(f"Using data type: {dtype}")

    crop_size = cfgs_data.get('crop_size', 224)
    patch_size = cfgs_data.get('patch_size')
    num_frames = cfgs_data.get('num_frames')  # Frames model expects per clip
    tubelet_size = cfgs_data.get('tubelet_size')
    sampling_rate = cfgs_data.get('sampling_rate', 4)
    duration = cfgs_data.get('clip_duration')
    num_clips = cfgs_data.get('num_clips', 1)
    decode_one_clip = cfgs_data.get('decode_one_clip', True)
    mask_type = cfgs_data.get('mask_type', 'multiblock3d')
    dataset_type = cfgs_data.get('dataset_type', 'videodataset')
    pin_mem = cfgs_data.get('pin_mem', False)

    model_name = cfgs_model.get('model_name')
    pred_depth = cfgs_model.get('pred_depth')
    pred_embed_dim = cfgs_model.get('pred_embed_dim')
    use_mask_tokens = cfgs_model.get('use_mask_tokens', True)
    zero_init_mask_tokens = cfgs_model.get('zero_init_mask_tokens', True)
    uniform_power = cfgs_model.get('uniform_power', True)
    use_sdpa = cfgs_meta.get('use_sdpa', False)

    # Data Aug settings
    ar_range = cfgs_data_aug.get('random_resize_aspect_ratio', [1.0, 1.0])
    rr_scale = cfgs_data_aug.get('random_resize_scale', [1.0, 1.0])
    motion_shift = False
    reprob = 0.0
    use_aa = False

    # --- 2. Setup Device ---
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load SSv2 labels
    ssv2_labels = load_ssv2_labels()
    logger.info(f"Loaded {len(ssv2_labels)} SSv2 labels")

    # --- 3. Check video file first ---
    logger.info(f"Checking video file: {args.video_path}")
    if not check_video_file(args.video_path):
        sys.exit(1)

    # --- 4. Initialize Model ---
    logger.info(f"Initializing model: {model_name}")
    encoder, predictor = init_video_model(
        uniform_power=uniform_power, use_mask_tokens=use_mask_tokens,
        num_mask_tokens=len(cfgs_mask), zero_init_mask_tokens=zero_init_mask_tokens,
        device='cpu', patch_size=patch_size, num_frames=num_frames, tubelet_size=tubelet_size,
        model_name=model_name, crop_size=crop_size, pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim, use_sdpa=use_sdpa)
    target_encoder = copy.deepcopy(encoder)

    # --- 5. Load Pretrained Weights ---
    logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Checkpoint file not found: {args.checkpoint_path}")
        sys.exit(1)
    
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        logger.info(f"Checkpoint dictionary keys found: {list(checkpoint.keys())}")

        if 'encoder' not in checkpoint:
            raise KeyError("Checkpoint missing 'encoder' state_dict.")
        if 'predictor' not in checkpoint:
            raise KeyError("Checkpoint missing 'predictor' state_dict.")
        
        target_encoder_state_dict_loaded = checkpoint.get('target_encoder', checkpoint['encoder'])

        encoder_state_dict_loaded = checkpoint['encoder']
        predictor_state_dict_loaded = checkpoint['predictor']

        encoder_state_dict_cleaned = {k.replace('module.', '', 1): v for k, v in encoder_state_dict_loaded.items()}
        predictor_state_dict_cleaned = {k.replace('module.', '', 1): v for k, v in predictor_state_dict_loaded.items()}
        target_encoder_state_dict_cleaned = {k.replace('module.', '', 1): v for k, v in target_encoder_state_dict_loaded.items()}

        missing_keys_enc, _ = encoder.load_state_dict(encoder_state_dict_cleaned, strict=False)
        missing_keys_pred, _ = predictor.load_state_dict(predictor_state_dict_cleaned, strict=False)
        missing_keys_target, _ = target_encoder.load_state_dict(target_encoder_state_dict_cleaned, strict=False)
        
        logger.info(f"Encoder missing keys: {len(missing_keys_enc)}")
        logger.info(f"Predictor missing keys: {len(missing_keys_pred)}")
        logger.info(f"Target Encoder missing keys: {len(missing_keys_target)}")

    except Exception as e:
        logger.error(f"Error loading checkpoint weights from {args.checkpoint_path}: {e}", exc_info=True)
        sys.exit(1)

    encoder.to(device).eval()
    predictor.to(device).eval()
    target_encoder.to(device).eval()
    for param in target_encoder.parameters():
        param.requires_grad = False

    # Initialize Pixel Decoder
    pixel_decoder = PixelDecoder(
        embed_dim=1024,  # Should match encoder output dimension
        decoder_embed_dim=512,
        patch_size=patch_size,
        num_frames=num_frames,
        crop_size=crop_size
    ).to(device)

    logger.info("Models initialized and weights loaded successfully.")

    # Load Attentive Probe
    attentive_probe = None
    if args.probe_path:
        attentive_probe = load_attentive_probe(
            probe_path=args.probe_path,
            embed_dim=1024,  # Should match base model output dimension
            num_classes=174,  # SSv2 has 174 classes
            device=device
        )

    # --- 6. Prepare Transforms and Mask Collator ---
    transform = make_transforms(
        random_horizontal_flip=False, random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale, reprob=reprob, auto_augment=use_aa,
        motion_shift=motion_shift, crop_size=crop_size)
    logger.info("Transforms prepared.")

    # Initialize Mask Collator
    MaskCollatorClass = MB3DMaskCollator if mask_type == 'multiblock3d' else TubeMaskCollator
    mask_collator = MaskCollatorClass(
        crop_size=crop_size, num_frames=num_frames, patch_size=patch_size,
        tubelet_size=tubelet_size, cfgs_mask=cfgs_mask)
    logger.info(f"Mask generator ({mask_type}) prepared.")

    # Setup DataLoader for Single Video
    logger.info(f"Setting up DataLoader for single video: {args.video_path}")
    video_abs_path = os.path.abspath(args.video_path)

    # Add this before the DataLoader setup
    import cv2
    cap = cv2.VideoCapture(args.video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    logger.info(f"Video info - Total frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
    logger.info(f"Required frames: {num_frames * sampling_rate}")

    if total_frames < num_frames * sampling_rate:
      logger.error(f"Video too short! Has {total_frames} frames, needs at least {num_frames * sampling_rate}")
      sys.exit(1)

    # --- 7. Setup DataLoader for Single Video ---
    logger.info(f"Setting up DataLoader for single video: {args.video_path}")
    video_abs_path = os.path.abspath(args.video_path)
    
    if not os.path.exists(video_abs_path):
        logger.error(f"Video file not found at resolved path: {video_abs_path}")
        sys.exit(1)

    # Create a temporary CSV file
    temp_csv_file = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_csv_file = f.name
            writer = csv.writer(f, delimiter=" ")
            writer.writerow([video_abs_path, 0])
        logger.info(f"Created temporary dataset CSV: {temp_csv_file}")
        
        # Debug: print CSV content
        with open(temp_csv_file, 'r') as f:
            logger.info(f"CSV content: {f.read().strip()}")

        # Initialize DataLoader - FIXED: Use the temporary CSV file path, not the video path list
        from src.datasets.video_dataset import make_videodataset

        dataset, dataloader, dist_sampler = make_videodataset(
        data_paths=[temp_csv_file],
        batch_size=1,
        frames_per_clip=num_frames,
        frame_step=sampling_rate,
        num_clips=num_clips,
        random_clip_sampling=False,  # Use sequential sampling for inference
        allow_clip_overlap=True,     # Allow overlap for short videos
        filter_short_videos=False,
        transform=transform,
        collator=mask_collator,
        num_workers=min(2, args.num_workers),
        pin_mem=pin_mem,
        duration=duration
        )
        logger.info("DataLoader initialized for the video.")

        # --- 8. Load Data and Masks from DataLoader ---
        logger.info("Loading data and generating masks via DataLoader...")
        
        # Test the dataloader
        dataloader_iter = iter(dataloader)
        
        # Add timeout and retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                loaded_batch = next(dataloader_iter)
                # DEBUG: Print the exact structure
                logger.info("=== BATCH STRUCTURE DEBUG ===")
                for i, item in enumerate(loaded_batch):
                    if isinstance(item, (list, tuple)):
                        logger.info(f"Batch[{i}]: {type(item)} with {len(item)} elements")
                        for j, subitem in enumerate(item):
                            if isinstance(subitem, (list, tuple)):
                                logger.info(f"  -> [{j}]: {type(subitem)} with {len(subitem)} elements")
                                for k, subsubitem in enumerate(subitem):
                                    logger.info(f"    -> [{j}][{k}]: {type(subsubitem)} - shape: {getattr(subsubitem, 'shape', 'No shape')}")
                            else:
                                logger.info(f"  -> [{j}]: {type(subitem)} - shape: {getattr(subitem, 'shape', 'No shape')}")
                    else:
                        logger.info(f"Batch[{i}]: {type(item)} - shape: {getattr(item, 'shape', 'No shape')}")
                logger.info("=== END DEBUG ===")
                logger.info("Successfully loaded batch from DataLoader")
                break
            except StopIteration:
                if attempt < max_retries - 1:
                    logger.warning(f"DataLoader empty, retrying... ({attempt + 1}/{max_retries})")
                    time.sleep(1)
                    dataloader_iter = iter(dataloader)  # Reset iterator
                else:
                    logger.error("DataLoader failed to yield any data after all retries.")
                    
                    # Additional debugging: try to inspect the dataset directly
                    try:
                        from src.datasets.video_dataset import VideoDataset
                        dataset = VideoDataset(
                            root_path=temp_csv_file,
                            clip_len=num_frames,
                            frame_sample_rate=sampling_rate,
                            transform=transform,
                            num_clips=num_clips,
                            duration=duration,
                            decode_one_clip=decode_one_clip
                        )
                        logger.info(f"Direct dataset inspection: {len(dataset)} samples")
                        if len(dataset) > 0:
                            sample = dataset[0]
                            logger.info(f"Sample shape: {sample.shape if hasattr(sample, 'shape') else 'No shape'}")
                    except Exception as e:
                        logger.error(f"Could not inspect dataset directly: {e}")
                    
                    raise StopIteration("DataLoader is empty")

        # Extract data and masks
        video_tensor_list = loaded_batch[0]
        masks_enc = loaded_batch[1]
        masks_pred = loaded_batch[2]

        logger.info("LEN VIDEO TENSOR LIST "+str(len(video_tensor_list)))
        video_tensor_list = video_tensor_list[0]
        if not isinstance(video_tensor_list, list) or len(video_tensor_list) != 1:
            logger.warning(f"Unexpected video tensor format from loader: {type(video_tensor_list)}. Expected list of length 1.")
            if isinstance(video_tensor_list, torch.Tensor) and video_tensor_list.ndim == 5:
                video_tensor = video_tensor_list
            else:
                raise ValueError("Cannot extract video tensor from DataLoader output.")
        else:
            video_tensor = video_tensor_list[0]

        # Ensure tensor and masks are on the correct device
        video_tensor = video_tensor.to(device)
        masks_enc = [m.to(device) for m in masks_enc]
        masks_pred = [m.to(device) for m in masks_pred]

        logger.info(f"Data loaded. Video tensor shape: {video_tensor.shape}")
        logger.info(f"Generated {len(masks_enc)} encoder masks and {len(masks_pred)} predictor masks.")

    except StopIteration as e:
        logger.error(f"DataLoader failed to yield any data: {e}")
        logger.error("This usually means:")
        logger.error("1. Video file is corrupted or in unsupported format")
        logger.error("2. Video is too short for the required clip length")
        logger.error(f"3. Required frames: {num_frames} with sampling rate: {sampling_rate}")
        logger.error("4. Check that OpenCV or decord can read your video file")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during data loading: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up the temporary CSV file
        if temp_csv_file and os.path.exists(temp_csv_file):
            os.remove(temp_csv_file)
            logger.info(f"Removed temporary dataset CSV: {temp_csv_file}")

    # --- 9. Perform Inference ---
    logger.info("Running model inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype, enabled=(dtype != torch.float32)):
            # Step 1: Get target features 'h'
            h = target_encoder(video_tensor)
            h = F.layer_norm(h, (h.size(-1),))
            h = apply_masks(h, masks_pred, concat=False)

            # Step 2: Get context features 'z_context'
            z_context = encoder(video_tensor, masks_enc)

            # Step 3: Get latent predictions 'z'
            z = predictor(z_context, h, masks_enc, masks_pred)

            # Step 4: Decode latent predictions to pixels
            predicted_frames = pixel_decoder(z, masks_enc, masks_pred)

    logger.info(f"Inference complete. Got {len(z)} latent prediction tensors.")


    # Run attentive probe if available
    probe_results = None
    if attentive_probe:
        logger.info("Running attentive probe for action classification...")
        
        # Prepare features for the probe
        # We can use different features: context features, target features, or both
        probe_features = []
        
        # Use context features (if available)
        if z_context is not None:
            if isinstance(z_context, list):
                for z_ctx in z_context:
                    probe_features.append(z_ctx)
            else:
                probe_features.append(z_context)
        
        # Use target features (if available)
        if h and len(h) > 0:
            for h_i in h:
                probe_features.append(h_i)
        
        # Run probe on all collected features
        if probe_features:
            probe_results = run_attentive_probe(attentive_probe, probe_features, ssv2_labels, top_k=5)
            logger.info(f"Probe analysis completed with {len(probe_results)} results")
        else:
            logger.warning("No suitable features found for probe analysis")

    logger.info(f"z_context type: {type(z_context)}")
    if isinstance(z_context, list):
        logger.info(f"z_context length: {len(z_context)}")
        for i, z_ctx in enumerate(z_context):
            logger.info(f"z_context[{i}] shape: {z_ctx.shape}")
    else:
        logger.info(f"z_context shape: {z_context.shape}")

    logger.info(f"h type: {type(h)}, length: {len(h)}")
    logger.info(f"z type: {type(z)}, length: {len(z)}")

    print_model_results(video_tensor, z_context, h, z, masks_enc, masks_pred, args.video_path,probe_results)

    # Plot representations
    representation_output = args.output_image.replace('.png', '_representations.png') if args.output_image else None
    plot_representations(z_context, h, z, masks_enc, masks_pred, representation_output)

    # 10 Viz
    logger.info("Preparing visualization...")
    num_vis_frames = min(args.num_vis, num_frames)

    # Get original frames for display
    # video_tensor shape: [1, 3, 16, 224, 224] = [batch, channels, frames, height, width]
    # We need to extract individual frames: [channels, height, width] for each frame

    # Method 1: Permute to [batch, frames, channels, height, width] then extract
    video_tensor_vis = video_tensor.permute(0, 2, 1, 3, 4)  # [1, 16, 3, 224, 224]
    original_frames_display = [unnormalize(video_tensor_vis[0, i]) for i in range(num_vis_frames)]

    # Or Method 2: Direct indexing (more explicit)
    # original_frames_display = []
    # for i in range(num_vis_frames):
    #     # Extract frame i: [3, 224, 224]
    #     frame = video_tensor[0, :, i, :, :]  # [channels, height, width]
    #     original_frames_display.append(unnormalize(frame))

    logger.info(f"Prepared {len(original_frames_display)} frames for display")

    # Get predicted frames for display
    predicted_frames_vis = predicted_frames.permute(0, 2, 1, 3, 4) if predicted_frames.shape[1] == 3 else predicted_frames
    predicted_frames_display = [unnormalize(predicted_frames_vis[0, i]) for i in range(num_vis_frames)]

    # Create masked input visualization (using placeholder/approximate logic as before)
    masked_input_frames_display = []
    try:
        patches_per_frame = (crop_size // patch_size) ** 2
        num_temporal_patches = num_frames // tubelet_size
        total_patches = patches_per_frame * num_temporal_patches
        
        # Use first encoder mask to estimate ratio
        if masks_enc and masks_enc[0] is not None and total_patches > 0:
            mask_ratio = 1.0 - (masks_enc[0].shape[-1] / total_patches)
        else:
            mask_ratio = 0.5
            
        logger.info(f"Approximate visual masking ratio: {mask_ratio:.2f}")

        for frame_pil in original_frames_display:
            frame_np = np.array(frame_pil).copy()
            h, w, _ = frame_np.shape
            ph, pw = patch_size, patch_size
            for r in range(0, h, ph):
                for c in range(0, w, pw):
                    if np.random.rand() < mask_ratio:
                        frame_np[r:r+ph, c:c+pw, :] = 0 # Black out patch
            masked_input_frames_display.append(Image.fromarray(frame_np))
    except Exception as e:
        logger.warning(f"Could not generate masked input visualization accurately: {e}. Using simple overlay.")
        masked_input_frames_display = [img.copy() for img in original_frames_display]

    # Plotting frames
    fig, axes = plt.subplots(3, num_vis_frames, figsize=(num_vis_frames * 2, 6))
    fig.suptitle(f"VJEPA Inference with Pixel Decoder: {os.path.basename(args.video_path)}")


    # --- Predicted Pixels (Placeholder) ---
    logger.warning("--------------------------------------------------------------------")
    logger.warning("Model produced LATENT predictions (variable 'z').")
    logger.warning("Visualizing these requires a Pixel Decoder specific to this model.")
    logger.warning("The 'Predicted' row below will show placeholders.")
    logger.warning("--------------------------------------------------------------------")

    # --- Plotting ---
    fig, axes = plt.subplots(3, num_vis_frames, figsize=(num_vis_frames * 2, 6))
    fig.suptitle(f"VJEPA Inference: {os.path.basename(args.video_path)}")

    for i in range(num_vis_frames):
        ax = axes[0, i]  # Original
        if i < len(original_frames_display):
            ax.imshow(original_frames_display[i])
        ax.set_title(f"Frame {i}")
        ax.axis('off')

        ax = axes[1, i]  # Masked Input
        if i < len(masked_input_frames_display):
            ax.imshow(masked_input_frames_display[i])
        ax.set_title(f"Masked In {i}")
        ax.axis('off')

        ax = axes[2, i]  # Predicted (Placeholder)
        placeholder_img = Image.new('RGB', original_frames_display[0].size, (50, 50, 50))
        ax.imshow(placeholder_img)
        ax.set_title(f"Pred {i} (N/A)")
        ax.axis('off')

    axes[0, 0].set_ylabel("Original", rotation=90, labelpad=20, verticalalignment='center', fontsize=10)
    axes[1, 0].set_ylabel("Masked Input\n(Approx.)", rotation=90, labelpad=20, verticalalignment='center', fontsize=10)
    axes[2, 0].set_ylabel("Predicted\n(Requires Decoder)", rotation=90, labelpad=20, verticalalignment='center', fontsize=10)

    plt.tight_layout(rect=[0.02, 0, 1, 0.95])
    if args.output_image:
        plt.savefig(args.output_image)
        logger.info(f"Visualization saved to {args.output_image}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VJEPA Inference Script using DataLoader')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file (e.g., .mp4)')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the pretrained model checkpoint file (e.g., .pth.tar)')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the model/data configuration YAML file')
    parser.add_argument('--output_image', type=str, default=None, help='Path to save the output visualization image (optional)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for inference')
    parser.add_argument('--num_vis', type=int, default=8, help='Number of frames to visualize in the output image')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of CPU workers for data loading')
    parser.add_argument('--probe_path', type=str, default=None, help='Path to the attentive probe checkpoint')

    args = parser.parse_args()

    # Basic validation
    if not os.path.isfile(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        sys.exit(1)
    if not os.path.isfile(args.config_path):
        print(f"Error: Config file not found at {args.config_path}")
        sys.exit(1)

    main_inference(args)
