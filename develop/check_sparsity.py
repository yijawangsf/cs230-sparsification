"""script to check the sparsity of obtained task vectors"""
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", None)

from src.task_vectors import LinearizedTaskVector


def check_effective_sparse_rate(tensor):
    """check achieved sparse rate of a tensor"""
    num_elements = tensor.ravel().size()[0]
    effective_sparse_rate = (tensor == 0).sum() / num_elements
    return effective_sparse_rate


def check_sparse_rate(path_model0, path_model1):
    """check achieved sparse rate of implicit task vector between 2 model checkpoints"""
    print(f"checking {checkpoint_dir}")
    task_vector = LinearizedTaskVector(path_model0, path_model1)
    effective_sparse_rate = _check_sparse_rate(task_vector)
    return effective_sparse_rate, task_vector


def _check_sparse_rate(task_vector):
    """check the portion of non-sparse tensors of a task_vector and the effective sparse rate of whole task vector"""
    non_sparse_vectors = set()
    sparse_rates = []
    sparse_counts = 0
    total_counts = 0
    for key, val in task_vector.vector.items():
        val_sum = val.sum()
        if val_sum != 0:
            # print(key, val_sum)
            non_sparse_vectors.add(key)
        sparse_rates.append(check_effective_sparse_rate(val))
        sparse_counts += (val == 0).sum()
        total_counts += val.ravel().size()[0]
    effective_sparse_rate = sparse_counts / total_counts
    print(f"non sparse vectors count {len(non_sparse_vectors) / len(task_vector.vector)}")
    print(f"effective sparse rate {effective_sparse_rate}")
    return effective_sparse_rate


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir_root", type=str)
    parsed_args = parser.parse_args()

    checkpoint_dir_root = parsed_args.checkpoint_dir_root


    intended_sparse_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    effective_sparse_rates_after = []
    effective_sparse_rates_before = []
    effective_sparse_rates_increase = []
    for rate in intended_sparse_rates:
        print(f"sparse rate = {rate}", "=" * 30)
        checkpoint_dir = f"{checkpoint_dir_root}/sparsify={rate}"
        path_model0 = f"{checkpoint_dir}/linear_zeroshot.pt"
        path_model1 = f"{checkpoint_dir}/linear_finetuned.pt"
        effective_sparse_rate_after, task_vector_after = check_sparse_rate(path_model0, path_model1)
        effective_sparse_rates_after.append(effective_sparse_rate_after.item())

        path_model2 = f"{checkpoint_dir}/linear_finetuned_backup.pt"
        effective_sparse_rate_before, task_vector_before = check_sparse_rate(path_model0, path_model2)
        effective_sparse_rates_before.append(effective_sparse_rate_before.item())

        # check the increase in sparsity
        sparse_increase = 0
        total_sparse_before = 0
        for param_name in task_vector_after.vector:
            vector_before = task_vector_before.vector[param_name]
            vector_after = task_vector_after.vector[param_name]
            sparse_before = (vector_before == 0).sum()
            sparse_after = (vector_after == 0).sum()
            total_sparse_before += sparse_before
            sparse_increase += (sparse_after - sparse_before)

        effective_sparse_rates_increase.append((sparse_increase / total_sparse_before).item())


    sparse_rates = {"intended_sparse_rates": intended_sparse_rates,  # intended sparse ratio
                    "effective_sparse_rates_after": effective_sparse_rates_after,  # effective abs sparse rates, after sparsify
                    "effective_sparse_rates_before": effective_sparse_rates_before,  # abs sparse rate, before sparsify
                    "effective_sparse_rates_increase": effective_sparse_rates_increase,}  # increase in sparsity (new sparsity - old sparsity) / old_sparsity
    sparse_rates = pd.DataFrame(sparse_rates)
    sparse_rates["effective_sparse_rate"] = (
            sparse_rates["effective_sparse_rates_before"] / sparse_rates["effective_sparse_rates_after"]
    )

    print(sparse_rates)


    print("complete")