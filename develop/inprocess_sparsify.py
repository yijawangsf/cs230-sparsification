"""
the script to conduct in-process sparsification
adapted from https://github.com/rezazzr/breadcrumbs/blob/main/src/task_vectors.py
"""
import torch


def mask_keep_top(tensor: torch.Tensor, top_k_keep: float=0) -> torch.Tensor:
    if len(tensor.shape) == 0:
        return tensor
    else:
        top_k_int = int(tensor.shape[-1] * top_k_keep)
        _, masked_indices = torch.topk(torch.abs(tensor), top_k_int)
        mask = torch.zeros(tensor.shape)
        mask.scatter_(len(tensor.shape) - 1, masked_indices, 1)
        return mask * tensor


def mask_remove_top(tensor: torch.Tensor, top_k_remove: float=0) -> torch.Tensor:
    if len(tensor.shape) == 0:
        return tensor
    else:
        top_k_int = int(tensor.shape[-1] * top_k_remove)
        _, masked_indices = torch.topk(torch.abs(tensor), top_k_int)
        mask = torch.ones(tensor.shape)
        mask.scatter_(len(tensor.shape) - 1, masked_indices, 0.0)
        return mask * tensor


def keep_topk_per_row(tensor, k):
    # Apply topk operation to each row
    values, indices = torch.topk(tensor, k, dim=1)
    mask = torch.zeros_like(tensor).scatter(1, indices, 1)
    return tensor * mask


def discard_topk_per_row(tensor, k):
    # Apply topk operation to each row and create a mask for discarding
    values, indices = torch.topk(tensor, k, dim=1)
    mask = torch.ones_like(tensor).scatter(1, indices, 0)
    return tensor * mask


if __name__ == "__main__":
    import argparse
    import shutil
    from src.task_vectors import LinearizedTaskVector


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
    )
    parser.add_argument(
        "--keep_rate",
        type=float,
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    pretrained_checkpoint = f"{checkpoint_dir}/linear_zeroshot.pt"
    finetuned_checkpoint = f"{checkpoint_dir}/linear_finetuned.pt"
    task_vector = LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)

    remove_first = True
    with torch.no_grad():
        for key, value in task_vector.vector.items():
            if remove_first:
                task_vector.vector[key] = mask_keep_top(mask_remove_top(value, 0.01), args.keep_rate)
            else:
                task_vector.vector[key] = mask_remove_top(mask_keep_top(value))

    image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)

    shutil.move(finetuned_checkpoint, finetuned_checkpoint.replace(".pt", "_backup.pt"))

    image_encoder.save(finetuned_checkpoint)

    print("complete")