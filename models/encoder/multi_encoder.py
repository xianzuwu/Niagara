import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision.transforms import ToPILImage
from models.encoder.resnet_encoder import ResnetEncoder
from models.decoder.resnet_decoder import ResnetDecoder, ResnetDepthDecoder, ResnetNormalDecoder
from PIL import Image
import torchvision.transforms as transforms
from .attention import SpatialTransformer3D

# Define TriplaneLearnablePositionalEmbedding class
class TriplaneLearnablePositionalEmbedding(nn.Module):
    def __init__(self, plane_size: int = 32, num_channels: int = 1024):
        super(TriplaneLearnablePositionalEmbedding, self).__init__()
        self.plane_size = plane_size
        self.num_channels = num_channels

        # Initial parameter
        self.embeddings = nn.Parameter(
            torch.randn(
                (3, self.num_channels, self.plane_size, self.plane_size),
                dtype=torch.float32,
            ) * 1 / math.sqrt(self.num_channels)
        )

    def forward(self, batch_size: int, cond_embeddings: torch.Tensor = None) -> torch.Tensor:
        embeddings = repeat(self.embeddings, "Np Ct Hp Wp -> B Np Ct Hp Wp", B=batch_size)
        if cond_embeddings is not None:
            cond = cond_embeddings.unsqueeze(-1).unsqueeze(-1)  # [B, Ct, 1, 1]
            embeddings = embeddings + cond
        return rearrange(embeddings, "B Np Ct Hp Wp -> B Ct (Np Hp Wp)")

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, Ct, Nt = tokens.shape
        assert Nt == self.plane_size**2 * 3, "Nt 不匹配 plane_size 和 Np"

        return rearrange(
            tokens, "B Ct (Np Hp Wp) -> B Np Ct Hp Wp", Np=3, Hp=self.plane_size, Wp=self.plane_size
        )

# Define new UniDepthExtended class，include GAF and Normal.
class UniDepthExtended(nn.Module):
    def __init__(self, cfg):
        super(UniDepthExtended, self).__init__()

        self.cfg = cfg
        self.devices = cfg.train.num_gpus
        
        # online
        # self.unidepth = torch.hub.load(
        #     "lpiccinelli-eth/UniDepth", "UniDepth", version=cfg.model.depth.version,
        #     backbone=cfg.model.depth.backbone, pretrained=True, trust_repo=True, force_reload=True
        # )
        
        # # outline
        self.unidepth = torch.hub.load(
            "lpiccinelli-eth/UniDepth", "UniDepth", version=cfg.model.depth.version,
            backbone=cfg.model.depth.backbone, pretrained=True, trust_repo=False, force_reload=False
        )

        self.normal_predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)
        
        self.parameters_to_train = []

        if cfg.model.backbone.name == "resnet":
            self.encoder = ResnetEncoder(
                num_layers=cfg.model.backbone.num_layers,
                pretrained=cfg.model.backbone.weights_init == "pretrained",
                bn_order=cfg.model.backbone.resnet_bn_order,\
            )
            #  3D Attetion
            actual_encoder_channels = self.encoder.num_ch_enc[-1]
            self.attention_3d = SpatialTransformer3D(
            in_channels=actual_encoder_channels,  
            n_heads=cfg.model.attn_heads,
            d_head=cfg.model.attn_dim_head,
            depth=cfg.model.attn_layers,
            context_dim=cfg.model.context_dim,
            disable_self_attn=cfg.model.disable_self_attn,
            use_checkpoint=cfg.model.use_checkpoint
        )
            self.parameters_to_train += [{"params": self.attention_3d.parameters()}]

            input_channels = 3  # RGB 
            if cfg.model.backbone.depth_cond:
                input_channels += 1  # Add Depth map
            input_channels += 3  # add normal map
            input_channels += 3 * cfg.triplane.num_channels  # Add triplane channel

            self.encoder.encoder.conv1 = nn.Conv2d(
                input_channels,
                self.encoder.encoder.conv1.out_channels,
                kernel_size=self.encoder.encoder.conv1.kernel_size,
                padding=self.encoder.encoder.conv1.padding,
                stride=self.encoder.encoder.conv1.stride
            )
            self.encoder.encoder.conv1.in_channels = input_channels
            self.parameters_to_train += [{"params": self.encoder.parameters()}]

            models = {}
            if cfg.model.gaussians_per_pixel > 1:
                models["depth"] = ResnetDepthDecoder(cfg=cfg, num_ch_enc=self.encoder.num_ch_enc)
                models["normal"] = ResnetNormalDecoder(cfg=cfg, num_ch_enc=self.encoder.num_ch_enc)
                self.parameters_to_train += [{"params": models["depth"].parameters()}]
                self.parameters_to_train += [{"params": models["normal"].parameters()}]
            for i in range(cfg.model.gaussians_per_pixel):
                models[f"gauss_decoder_{i}"] = ResnetDecoder(cfg=cfg, num_ch_enc=self.encoder.num_ch_enc)
                self.parameters_to_train += [{"params": models[f"gauss_decoder_{i}"].parameters()}]
                if cfg.model.one_gauss_decoder:
                    break
            self.models = nn.ModuleDict(models)

        self.triplane = TriplaneLearnablePositionalEmbedding(
            plane_size=cfg.triplane.plane_size,
            num_channels=cfg.triplane.num_channels
        )
        self.parameters_to_train += [{"params": self.triplane.parameters()}]

        self.to_pil = transforms.ToPILImage()  
        self.to_tensor = transforms.ToTensor()  

    def get_parameter_groups(self):
        return self.parameters_to_train
            
    def forward(self, inputs):
        # If there is no depth in the input, use the pre-trained model to predict the depth
        
        # print(f"Forward inputs: {inputs}")
        if ('unidepth', 0, 0) in inputs.keys() and inputs[('unidepth', 0, 0)] is not None:
            depth_outs = dict()
            depth_outs = {"depth": inputs[('unidepth', 0, 0)]}
            # print(depth_outs)
        else:
            with torch.no_grad():
                intrinsics = inputs.get(("K_src", 0), None)
                depth_outs = self.unidepth.infer(inputs["color_aug", 0, 0], intrinsics=intrinsics)
            # h, w = inputs[“color_aug”, 0, 0].shape[2:]
            # # # Create an all-zero or all-one depth tensor as the default value
            #     default_depth = torch.ones((inputs["color_aug", 0, 0].shape[0], 1, h, w)).to(inputs["color_aug", 0, 0].device)
            #     depth_outs = {"depth": default_depth}
            # print("No unidepth data found, skipping depth processing.")
            
        outputs_gauss = {}
        outputs_gauss[("K_src", 0)] = inputs.get(("K_src", 0), depth_outs.get("intrinsics"))
        # outputs_gauss[("inv_K_src", 0)] = torch.linalg.inv(outputs_gauss[("K_src", 0)])
        outputs_gauss[("inv_K_src", 0)] = torch.linalg.inv(outputs_gauss[("K_src", 0)].float())

        # predict normal map
        normal_outs = {}
        if ('normal', 0, 0) in inputs.keys() and inputs[('normal', 0, 0)] is not None:
            normal_outs = dict()
            normal_outs["normal"] = inputs[('normal', 0, 0)]
        else:
            with torch.no_grad():
                normal_outs_tensors = []
                for i in range(inputs["color_aug", 0, 0].shape[0]):
                    image_pil = self.to_pil(inputs["color_aug", 0, 0][i].cpu())
                    normal_image_pil = self.normal_predictor(image_pil)
                    normal_tensor = self.to_tensor(normal_image_pil).to(inputs["color_aug", 0, 0].device)
                    normal_outs_tensors.append(normal_tensor)
                normal_outs["normal"] = torch.stack(normal_outs_tensors).to(inputs["color_aug", 0, 0].device)

        normal_outs["normal"] = (normal_outs["normal"] + 1) / 2.0

        # Conditional inputs for processing RGB images and depth and normal maps
        if self.cfg.model.backbone.depth_cond:
            depth_input = depth_outs["depth"] / 20.0
            normal_input = normal_outs["normal"]
            
            h, w = inputs["color_aug", 0, 0].shape[2:]
            depth_input = F.interpolate(depth_input, size=(h, w), mode='bilinear', align_corners=False)
            normal_input = F.interpolate(normal_input, size=(h, w), mode='bilinear', align_corners=False)

            input_rgb_depth_normal = torch.cat([inputs["color_aug", 0, 0], depth_input, normal_input], dim=1)
        else:
            input_rgb_depth_normal = inputs["color_aug", 0, 0]

        batch_size = input_rgb_depth_normal.size(0)
        triplane_embeddings = self.triplane(batch_size)
        triplane_reshaped = self.triplane.detokenize(triplane_embeddings)

        _, _, H, W = input_rgb_depth_normal.shape
        triplane_reshaped = triplane_reshaped.reshape(
            batch_size * 3, self.cfg.triplane.num_channels, self.cfg.triplane.plane_size, self.cfg.triplane.plane_size
        )
        triplane_features = F.interpolate(
            triplane_reshaped, size=(H, W), mode='bilinear', align_corners=False
        )

        triplane_features = triplane_features.view(batch_size, 3 * self.cfg.triplane.num_channels, H, W)
        combined_input = torch.cat([input_rgb_depth_normal, triplane_features], dim=1)
        encoded_features = self.encoder(combined_input)
        # 3D Self-Attention
        attn_features = self.attention_3d(
            encoded_features[-1],  
            num_frames=self.cfg.model.num_frames
        )
        
        # Merge the attention feature back into the original feature system
        encoded_features = list(encoded_features)
        encoded_features[-1] = attn_features
        encoded_features = tuple(encoded_features)
        outputs_gauss["encoded_features"] = encoded_features[-1]

        # predict multi-gaussian
        if self.cfg.model.gaussians_per_pixel > 1:
            depth = self.models["depth"](encoded_features)
            depth = rearrange(depth[("depth", 0)], "(b n) ... -> b n ...", n=self.cfg.model.gaussians_per_pixel - 1)
            depth = torch.cumsum(torch.cat((depth_outs["depth"][:, None, ...], depth), dim=1), dim=1)
            outputs_gauss[("depth", 0)] = rearrange(depth, "b n c ... -> (b n) c ...", n=self.cfg.model.gaussians_per_pixel)
        else:
            outputs_gauss[("depth", 0)] = depth_outs["depth"]

        # predict normal map
        if self.cfg.model.gaussians_per_pixel > 1:
            normal = self.models["normal"](encoded_features)
            normal_out_4d = normal_outs["normal"]
            normal_3d = normal[("normal", 0)]

            normal_4d = normal_3d.unsqueeze(1)

            # aline normal_out_4d, normal_4d 
            _, _, h, w = normal_out_4d.shape
            normal_4d = F.interpolate(normal_4d, size=(h, w), mode='bilinear', align_corners=False)

            # aline normal_out_4d, normal_4d batch
            if normal_out_4d.shape[0] != normal_4d.shape[0]:
                normal_out_4d = normal_out_4d.repeat(normal_4d.shape[0] // normal_out_4d.shape[0], 1, 1, 1)
            if normal_out_4d.shape[1] != normal_4d.shape[1]:
                normal_4d = normal_4d.repeat(1, normal_out_4d.shape[1] // normal_4d.shape[1], 1, 1)

            combined = torch.cat((normal_out_4d.unsqueeze(1), normal_4d.unsqueeze(1)), dim=1)
            normal[("normal", 0)] = torch.cumsum(combined, dim=1)
            outputs_gauss[("normal", 0)] = normal[("normal", 0)].view(-1, normal[("normal", 0)].shape[2], normal[("normal", 0)].shape[3], normal[("normal", 0)].shape[4])
        else:
            outputs_gauss[("normal", 0)] = normal_outs["normal"]

        # predict multi-gaussian
        gauss_outs = {}
        for i in range(self.cfg.model.gaussians_per_pixel):
            outs = self.models[f"gauss_decoder_{i}"](encoded_features)
            if self.cfg.model.one_gauss_decoder:
                gauss_outs |= outs
                break
            else:
                for key, v in outs.items():
                    if i == 0:
                        gauss_outs[key] = v
                    else:
                        gauss_outs[key] = torch.cat([gauss_outs[key], v], dim=1)
        for key, v in gauss_outs.items():
            gauss_outs[key] = rearrange(v, 'b n ... -> (b n) ...')
        outputs_gauss |= gauss_outs

        return outputs_gauss