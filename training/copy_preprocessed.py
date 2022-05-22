import argparse
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
import os
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_name", help="can be task name or task id")
    parser.add_argument("src_preprocessed_dir")
    parser.add_argument("dst_preprocessed_dir")
    args = parser.parse_args()

    task_name = args.task_name

    if not task_name.startswith("Task"):
        task_id = int(task_name)
        task_name = convert_id_to_task_name(task_id)

    src = os.path.join(args.src_preprocessed_dir, task_name)
    dst = os.path.join(args.dst_preprocessed_dir, task_name)

    os.makedirs(dst, exist_ok=True)
    print(f"Copying preprocessed data from {src} into container...")

    shutil.copytree(src, dst, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
