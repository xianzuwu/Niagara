import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from models.encoder.resnet_encoder import ResnetEncoder
from models.decoder.resnet_decoder import ResnetDecoder, ResnetNormalDecoder

class StableNormalExtended(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        # Load the pre-trained StableNormal model from Stable-X repository
        self.stablenormal = torch.hub.load(
            "Stable-X/StableNormal", "StableNormal", trust_repo=True
        )

        self.parameters_to_train = []
        if cfg.model.backbone.name == "resnet":
            # Initialize the ResNet encoder from the config settings
            self.encoder = ResnetEncoder(
                num_layers=cfg.model.backbone.num_layers,
                pretrained=cfg.model.backbone.weights_init == "pretrained",
                bn_order=cfg.model.backbone.resnet_bn_order,
            )
            
            # Add encoder parameters to the list of trainable parameters
            self.parameters_to_train += [{"params": self.encoder.parameters()}]

            # Initialize the normal decoder for multi-scale normal map prediction
            self.normal_decoder = ResnetNormalDecoder(cfg=cfg, num_ch_enc=self.encoder.num_ch_enc)
            self.parameters_to_train += [{"params": self.normal_decoder.parameters()}]

    def get_parameter_groups(self):
        # Return trainable parameters: encoder and normal decoder
        return self.parameters_to_train

    def forward(self, inputs):
        # Predict the normal map using the pre-trained StableNormal model
        if "stablenormal" in inputs.keys() and inputs["stablenormal"] is not None:
            normal_outs = dict()
            normal_outs["normal"] = inputs["stablenormal"]
        else:
            with torch.no_grad():
                input_image = inputs["color_aug", 0, 0]
                # Use StableNormal model to infer the normal map
                normal_outs = {"normal": self.stablenormal(input_image)}

        # Pass through the encoder and decoder for further refinement
        encoded_features = self.encoder(normal_outs["normal"])
        refined_normal = self.normal_decoder(encoded_features)

        # Combine the refined normal map with the original
        outputs = {}
        outputs["normal"] = refined_normal[("normal", 0)]  # You can handle multiple scales if necessary

        return outputs

# Example configuration for testing
if __name__ == "__main__":
    from PIL import Image
    cfg = {
        "model": {
            "backbone": {
                "name": "resnet",
                "num_layers": 18,
                "weights_init": "pretrained",
                "resnet_bn_order": "pre_bn",
            },
            "normal_scale": 0.1,
            "normal_bias": 0.0,
            "scales": [0],
        }
    }

    # Load a test image
    input_image = Image.open("path/to/your/image.jpg")

    # Create an instance of the StableNormalExtended model
    model = StableNormalExtended(cfg)

    # Create a mock input dictionary
    inputs = {"color_aug": {(0, 0): input_image}}

    # Run forward pass
    output = model(inputs)

    # Save or display the output normal map
    normal_map = output["normal"]
    normal_map.save("output/normal_map.png")
