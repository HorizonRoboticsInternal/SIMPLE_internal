#!/usr/bin/env python3
"""Small build/runtime check for the model-only HoloBrain environment."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imports-only", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--vlm-pretrain")
    parser.add_argument("--action-tokenizer-path")
    args = parser.parse_args()

    import fastapi
    import flash_attn
    import pytorch3d
    import robo_orchard_core
    import torch
    import transformers
    from scripts.holobrain_simple_policy import HoloBrainSimplePolicy

    print(
        "HOLOBRAIN_IMPORT_OK",
        f"torch={torch.__version__}",
        f"cuda={torch.version.cuda}",
        f"transformers={transformers.__version__}",
        f"fastapi={fastapi.__version__}",
        f"flash_attn={flash_attn.__version__}",
        f"pytorch3d={pytorch3d.__version__}",
        f"core={robo_orchard_core.__version__}",
    )

    if args.imports_only:
        return

    if not args.checkpoint or not args.vlm_pretrain or not args.action_tokenizer_path:
        parser.error(
            "--checkpoint, --vlm-pretrain, and --action-tokenizer-path are "
            "required unless --imports-only is used"
        )
    policy = HoloBrainSimplePolicy(
        checkpoint=args.checkpoint,
        vlm_pretrain=args.vlm_pretrain,
        action_tokenizer_path=args.action_tokenizer_path,
        device="cuda:0",
        action_exec_horizon=24,
    )
    print("HOLOBRAIN_MODEL_OK", policy.model_kind, type(policy.model).__name__)


if __name__ == "__main__":
    main()
