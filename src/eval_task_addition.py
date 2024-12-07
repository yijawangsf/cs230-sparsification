"""
adapted from
https://github.com/gortizji/tangent_task_arithmetic/blob/main/src/eval_task_addition.py
"""
import json
import os

from utils import find_optimal_coef

from src.args import parse_arguments
from src.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()

if args.seed is not None:
    args.save = f"checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"{args.checkpoint_dir}"


print("*" * 100)
if args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")
    ft_accuracies_path = os.path.join(args.save, "ft_accuracies.json")
elif args.finetuning_mode == "linear":
    print("Evaluating linear FT models.")
    ft_accuracies_path = os.path.join(args.save, "linear_ft_accuracies.json")
elif args.finetuning_mode == "posthoc":
    print("Evaluating post-hoc linearized models.")
    ft_accuracies_path = os.path.join(args.save, "posthoc_ft_accuracies.json")
else:
    raise ValueError(f"Invalid finetuning mode: {args.finetuning_mode}")
print("*" * 100)


# target datasets
eval_datasets = [
    "EuroSAT",
    "RESISC45",
    "Cars",
    "DTD",
]

# full eval datasets including hidden datasets
eval_datasets2 = [
    "EuroSAT",
    "RESISC45",
    "Cars",
    "DTD",
    "GTSRB",
    "MNIST",
    "SUN397",
    "SVHN",
]

task_vectors = []

for dataset in eval_datasets:
    if args.finetuning_mode == "linear":
        pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
        task_vectors.append(
            LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )
    else:
        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
        task_vectors.append(
            NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )

task_vector = sum(task_vectors)

args.eval_datasets = [dataset + "Val" for dataset in eval_datasets]
args.control_dataset = None

# We use the validation set to choose the optimal coefficient.
val_metrics = evaluate_task_vector(
    task_vector,
    pretrained_checkpoint,
    args,
    posthoc_linearization=args.finetuning_mode == "posthoc",
)

optimal_coef = find_optimal_coef(
    val_metrics,
    metric="avg_top1",
    minimize=False,
)

print(f"optimal_coef={optimal_coef}")

# Evaluate on the test set with the optimal coefficient.
args.eval_datasets = [dataset for dataset in eval_datasets2]
test_metrics = evaluate_task_vector_at_coef(
    task_vector,
    pretrained_checkpoint,
    args,
    float(optimal_coef),
    posthoc_linearization=args.finetuning_mode == "posthoc",
)

print("=" * 100)
additive_accuracies = {"test": test_metrics, "val": val_metrics}

if args.finetuning_mode == "standard":
    save_file = f"{args.save}/additions.json"
elif args.finetuning_mode == "linear":
    save_file = f"{args.save}/linear_additions.json"
elif args.finetuning_mode == "posthoc":
    save_file = f"{args.save}/posthoc_additions.json"
with open(save_file, "w") as f:
    json.dump(additive_accuracies, f, indent=4)
