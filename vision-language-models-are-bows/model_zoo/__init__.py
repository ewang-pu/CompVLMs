import os
import clip
from PIL import Image
from torchvision import transforms
from .constants import CACHE_DIR
from transformers import ViltProcessor, ViltForImageAndTextRetrieval, ViltImageProcessor


def get_model(model_name, device, root_dir=CACHE_DIR):
    """
    Helper function that returns a model and a potential image preprocessing function.
    """
    if "openai-clip" in model_name:
        from .clip_models import CLIPWrapper

        variant = model_name.split(":")[1]
        model, image_preprocess = clip.load(
            variant, device=device, download_root=root_dir
        )
        model = model.eval()
        clip_model = CLIPWrapper(model, device)
        return clip_model, image_preprocess

    elif "vilt" in model_name:
        from .vilt_models import ViLTWrapper

        processor = ViltProcessor.from_pretrained(
            os.path.join(root_dir, "../local_models/", model_name)
        )

        # image_processor = ViltImageProcessor.from_pretrained(
        #     "../model_zoo/vilt-b32-finetuned-coco"
        # )
        model = ViltForImageAndTextRetrieval.from_pretrained(
            os.path.join(root_dir, "../local_models/", model_name)
        )
        vilt_model = ViLTWrapper(model, processor, device=device)
        image_preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (384, 384),
                    interpolation=transforms.functional.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
            ]
        )

        return vilt_model, image_preprocess

    elif model_name == "blip-flickr-base":
        from .blip_models import BLIPModelWrapper

        blip_model = BLIPModelWrapper(
            root_dir=root_dir, device=device, variant="blip-flickr-base"
        )
        image_preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (384, 384),
                    interpolation=transforms.functional.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        return blip_model, image_preprocess

    elif model_name == "blip-coco-base":
        from .blip_models import BLIPModelWrapper

        blip_model = BLIPModelWrapper(
            root_dir=root_dir, device=device, variant="blip-coco-base"
        )
        image_preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (384, 384),
                    interpolation=transforms.functional.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        return blip_model, image_preprocess

    elif model_name == "xvlm-flickr":
        from .xvlm_models import XVLMWrapper

        xvlm_model = XVLMWrapper(
            root_dir=root_dir, device=device, variant="xvlm-flickr"
        )
        image_preprocess = transforms.Compose(
            [
                transforms.Resize((384, 384), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        return xvlm_model, image_preprocess

    elif model_name == "xvlm-coco":
        from .xvlm_models import XVLMWrapper

        xvlm_model = XVLMWrapper(root_dir=root_dir, device=device, variant="xvlm-coco")
        image_preprocess = transforms.Compose(
            [
                transforms.Resize((384, 384), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        return xvlm_model, image_preprocess

    elif model_name == "flava":
        from .flava import FlavaWrapper

        flava_model = FlavaWrapper(root_dir=root_dir, device=device)
        image_preprocess = None
        return flava_model, image_preprocess

    elif "NegCLIP" in model_name:
        import open_clip
        from .clip_models import CLIPWrapper

        if ":" in model_name:
            variant = model_name.split(":")[1]
            path = os.path.join(root_dir, "negCLIP_" + variant + ".pt")
        else:
            path = os.path.join(root_dir, "negCLIP.pt")
        if not os.path.exists(path):
            print("Downloading the NegCLIP model...")
            import gdown

            gdown.download(
                id="1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ", output=path, quiet=False
            )
        model, _, image_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=path, device=device
        )
        model = model.eval()
        clip_model = CLIPWrapper(model, device)
        return clip_model, image_preprocess

    elif model_name == "coca":
        import open_clip
        from .clip_models import CLIPWrapper

        model, _, image_preprocess = open_clip.create_model_and_transforms(
            model_name="coca_ViT-B-32", pretrained="laion2B-s13B-b90k", device=device
        )
        model = model.eval()
        clip_model = CLIPWrapper(model, device)
        return clip_model, image_preprocess

    elif "laion-clip" in model_name:
        import open_clip
        from .clip_models import CLIPWrapper

        variant = model_name.split(":")[1]
        model, _, image_preprocess = open_clip.create_model_and_transforms(
            model_name=variant, pretrained="laion2b_s34b_b79k", device=device
        )
        model = model.eval()
        clip_model = CLIPWrapper(model, device)
        return clip_model, image_preprocess

    else:
        raise ValueError(f"Unknown model {model_name}")
