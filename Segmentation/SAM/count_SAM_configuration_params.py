import os
import itertools
from pathlib import Path

from segment_anything import sam_model_registry
from Segmentation.SAM.sam_lora import SamLoRA, EncoderQKVLoRA, DecoderAttentionProjLoRA
from Environment.env_utils import load_forksight_env

load_forksight_env()

MODEL_CHECKPOINTS_DIR = os.getenv("MODEL_CHECKPOINTS_DIR")


def count_parameters(sam_lora: SamLoRA):
    components = {
        "image_encoder": sam_lora.sam_model.image_encoder,
        "prompt_encoder": sam_lora.sam_model.prompt_encoder,
        "mask_decoder": sam_lora.sam_model.mask_decoder,
    }

    lora_classes = (EncoderQKVLoRA, DecoderAttentionProjLoRA)
    wrapped_attr_names = {"qkv", "proj"}

    # Collect ids of parameters added by LoRA (i.e., not in the wrapped original).
    lora_param_ids = set()
    for module in sam_lora.modules():
        if isinstance(module, lora_classes):
            for child_name, child in module.named_children():
                if child_name in wrapped_attr_names:
                    continue
                for p in child.parameters():
                    lora_param_ids.add(id(p))

    results = {"trainable": {}, "total": {}, "original": {}}

    for name, module in components.items():
        trainable = total = original = 0
        for param in module.parameters():
            n = param.numel()
            total += n
            if param.requires_grad:
                trainable += n
            if id(param) not in lora_param_ids:
                original += n

        results["trainable"][name] = trainable
        results["total"][name] = total
        results["original"][name] = original

    for key in results:
        results[key]["all"] = sum(results[key].values())

    return results


def print_parameter_counts(sam_lora: SamLoRA):
    counts = count_parameters(sam_lora)
    header = f"{'Component':<20}{'Trainable':>15}{'Total':>15}{'Original SAM':>15}"
    print(header)
    print("-" * len(header))
    for name in ["image_encoder", "prompt_encoder", "mask_decoder", "all"]:
        print(f"{name:<20}"
              f"{counts['trainable'][name]:>15,}"
              f"{counts['total'][name]:>15,}"
              f"{counts['original'][name]:>15,}")


def main():
    sam_model_type = "vit_b"
    sam_checkpoint_name = "sam_vit_b_01ec64"

    sam_checkpoint = str(Path(MODEL_CHECKPOINTS_DIR) /
                         f"{sam_checkpoint_name}.pth")

    configurations = []

    for n_blocks, r in itertools.product([0, 1, 4], [4, 8]):
        if n_blocks > 0:
            # Full fine-tuning of last N blocks, no LoRA on encoder
            configurations.append({
                "finetune_img_encoder_lora": False,
                "finetune_img_encoder_n_blocks": n_blocks,
                "finetune_mask_decoder": True,
                "finetune_prompt_encoder": True,
                "r": r,
            })
        else:
            # LoRA on encoder (n_blocks=0)
            for lora in [True, False]:
                configurations.append({
                    "finetune_img_encoder_lora": lora,
                    "finetune_img_encoder_n_blocks": 0,
                    "finetune_mask_decoder": True,
                    "finetune_prompt_encoder": True,
                    "r": r,
                })

    for i, config in enumerate(configurations):
        # Reload fresh SAM each time since SamLoRA modifies it in-place
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)

        model = SamLoRA(
            sam_model=sam,
            r=config["r"],
            finetune_img_encoder_lora=config["finetune_img_encoder_lora"],
            finetune_img_encoder_n_blocks=config["finetune_img_encoder_n_blocks"],
            finetune_mask_decoder=config["finetune_mask_decoder"],
            finetune_prompt_encoder=config["finetune_prompt_encoder"],
        )

        desc = (f"n_blocks={config['finetune_img_encoder_n_blocks']}, "
                f"finetune_img_encoder_lora={config['finetune_img_encoder_lora']}, "
                f"r={config['r']}")

        print(f"\n{'=' * 70}")
        print(f"Config {i + 1}: {desc}")
        print(f"{'=' * 70}")
        print_parameter_counts(model)

        # Also print trainable as % of original
        counts = count_parameters(model)
        pct = counts["trainable"]["all"] / counts["original"]["all"] * 100
        print(f"{'Trainable %':<20}{pct:>15.2f}%")


if __name__ == "__main__":
    main()
