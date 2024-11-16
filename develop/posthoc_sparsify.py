"""script that conduct post-hoc sparsification"""
import os

import torch


def check_effective_sparse_rate(tensor):
    num_elements = tensor.ravel().size()[0]
    effective_sparse_rate = (tensor == 0).sum() / num_elements
    return effective_sparse_rate


def check_sparse_rate(tensor, mask):
    num_elements = tensor.ravel().size()[0]
    keep_rate = mask.sum() / num_elements
    result = mask * tensor
    effective_sparse_rate = check_effective_sparse_rate(result)
    print(
        f"sparse rate: intended={torch.round(100 * (1 - keep_rate))}%, "
        f"effective={torch.round(100 * effective_sparse_rate)}%")
    return


def mask_keep_top(tensor: torch.Tensor, top_k_keep: float=0) -> torch.Tensor:
    if len(tensor.shape) == 0:
        return tensor
    else:
        top_k_int = int(tensor.shape[-1] * top_k_keep)
        _, masked_indices = torch.topk(torch.abs(tensor), top_k_int)
        mask = torch.zeros(tensor.shape)
        mask.scatter_(len(tensor.shape) - 1, masked_indices, 1)

        # check_sparse_rate(tensor, mask)
        return mask * tensor


def get_size(tensor):
    if torch.is_tensor(tensor):
        return tensor.shape
    elif isinstance(tensor, (float, int)):
        return torch.Size([])  # Treat the scalar as an empty shape
    else:
        raise ValueError("Input must be a tensor, float, or int.")


def mask_remove_top(tensor: torch.Tensor, top_k_remove: float=0) -> torch.Tensor:
    if len(tensor.shape) == 0:
        return tensor
    else:
        top_k_int = int(tensor.shape[-1] * top_k_remove)
        _, masked_indices = torch.topk(torch.abs(tensor), top_k_int)
        mask = torch.ones(tensor.shape)
        mask.scatter_(len(tensor.shape) - 1, masked_indices, 0.0)

        # check_sparse_rate(tensor, mask)
        return mask * tensor


if __name__ == "__main__":
    import shutil
    from src.task_vectors import LinearizedTaskVector
    from paths import CODE_DIR


    root_dir = f"{CODE_DIR}/checkpoints-origseq"
    from_dir_pretrained = f"{root_dir}/EuroSATVal"
    from_dir_finetuned = f"{root_dir}/DTDVal"
    for sparse_rate in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f"rate={sparse_rate}")
        checkpoint_dir = f"{root_dir}/FullSparsified/sparsify={sparse_rate}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = f"{checkpoint_dir}/linear_zeroshot.pt"
        finetuned_checkpoint = f"{checkpoint_dir}/linear_finetuned.pt"

        shutil.copy(f"{from_dir_pretrained}/linear_zeroshot.pt", f"{checkpoint_dir}/linear_zeroshot.pt")
        shutil.copy(f"{from_dir_finetuned}/linear_finetuned.pt", f"{checkpoint_dir}/linear_finetuned.pt")

        pretrained_checkpoint_saveto = pretrained_checkpoint
        finetuned_checkpoint_saveto = finetuned_checkpoint

        task_vector = LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)

        remove_first = True
        with torch.no_grad():
            for key, value in task_vector.vector.items():
                if remove_first:
                    task_vector.vector[key] = mask_keep_top(mask_remove_top(value, 0.01), 1 - sparse_rate)
                else:
                    task_vector.vector[key] = mask_remove_top(mask_keep_top(value))

        image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)

        if finetuned_checkpoint_saveto == finetuned_checkpoint:
            shutil.move(finetuned_checkpoint, finetuned_checkpoint.replace(".pt", "_backup.pt"))
        image_encoder.save(finetuned_checkpoint_saveto)

        if pretrained_checkpoint_saveto != pretrained_checkpoint:
            shutil.copy(pretrained_checkpoint, pretrained_checkpoint_saveto)

        import subprocess

        # Path to the Python script
        script_path = f"{CODE_DIR}/src/eval_single_task.py"

        # Run the Python script with named arguments
        result = subprocess.run(
            ["python", script_path,
                    "--finetuning-mode", "linear",
                    "--checkpoint_dir", checkpoint_dir,
                    "--load", checkpoint_dir,
                    "--head_dir", f"{CODE_DIR}/classification_heads",
                    "--eval-datasets", "EuroSAT,Cars,RESISC45,DTD",
             ],
            capture_output=True,
            text=True
        )

        # Output from the script
        print("Return code:", result.returncode)
        print("Output:", result.stdout)
        print("Error:", result.stderr)

    print("complete")