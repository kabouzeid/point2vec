import torch


def extract_model_checkpoint(path: str):
    checkpoint = torch.load(path)

    if "state_dict" in checkpoint:
        # lightning checkpoint
        checkpoint = {
            k.replace("student.", "encoder.").replace(
                "pos_embedding.", "positional_encoding."
            ): v
            for k, v in checkpoint["state_dict"].items()
        }
        for k in list(checkpoint.keys()):
            if k.startswith("teacher."):
                del checkpoint[k]
            elif k.startswith("regressor."):
                del checkpoint[k]
            elif k.startswith("decoder."):
                del checkpoint[k]
            elif k == "mask_token":
                del checkpoint[k]
    elif "base_model" in checkpoint:
        # Point-MAE or Point-BERT
        if (
            "transformer_q.cls_token" in checkpoint["base_model"]
        ):  # Point-BERT pretrained
            checkpoint["base_model"] = {
                k.replace("transformer_q.", "module."): v
                for k, v in checkpoint["base_model"].items()
            }
            for k in list(checkpoint["base_model"].keys()):
                if not k.startswith("module."):
                    del checkpoint["base_model"][k]

        checkpoint = {
            k
            # Point-MAE
            .replace("module.MAE_encoder.blocks.", "encoder.")
            .replace("module.MAE_encoder.norm.", "encoder.norm.")
            .replace("module.MAE_encoder.pos_embed.", "positional_encoding.")
            .replace("module.MAE_encoder.encoder.", "tokenizer.embedding.")
            # Point-BERT
            .replace("module.encoder.", "tokenizer.embedding.")
            .replace("module.reduce_dim.", "tokenizer_to_encoder.")
            .replace("module.blocks.", "encoder.")
            .replace("module.norm.", "encoder.norm.")
            .replace("module.pos_embed.", "positional_encoding.")
            .replace("module.cls_token", "cls_token")
            .replace("module.cls_pos", "cls_pos")
            .replace("module.cls_head_finetune", "cls_head")
            # finally
            .replace("module.", ""): v
            for k, v in checkpoint["base_model"].items()
        }
        for k in list(checkpoint.keys()):
            if k.startswith("MAE_decoder."):
                del checkpoint[k]
            elif k == "mask_token":
                del checkpoint[k]
            elif k.startswith("decoder_pos_embed."):
                del checkpoint[k]
            elif k.startswith("increase_dim."):
                del checkpoint[k]
            elif k in ["cls_token", "cls_pos"]:
                checkpoint[k] = checkpoint[k].squeeze(0).squeeze(0)
            elif k.startswith("lm_head."):
                del checkpoint[k]
    else:
        raise RuntimeError("Unknown checkpoint format")

    return checkpoint
