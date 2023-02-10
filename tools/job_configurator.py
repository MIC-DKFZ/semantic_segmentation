import itertools
import numpy as np

if __name__ == "__main__":
    """
    Cluster Configuration
    """
    # Required
    queue = "gpu-lowprio.legacy"
    gmem = "18.0G"
    num_gpus = 1

    # Optional
    job_group = "/l727r/group_15"

    command = (
        f"bsub -gpu num={num_gpus}:j_exclusive=yes:mode=exclusive_process:gmem={gmem} -L /bin/bash"
        f" -q {queue}"
    )

    if job_group is not None:
        command += f" -g {job_group}"

    """
    Job Configuration
    """
    script = "training.py"
    bash = ".bashrc_sem_seg_2"

    job_config = f"source ~/{bash} && python3 {script}"

    """
    Configure Parameters
    """
    # Hard parameters - set to value in each run
    parameters_hard = [
        ("dataset", "COMPUTING_2"),
        ("model", "hrnet_ocr_ms"),
        ("environment", "cluster"),
    ]

    # Parameters to run in every combination
    parameters_all_of = [
        ("dataset.fold", [0, 1, 2, 3, 4]),
        ("dataset.random_sampling", [0.3, 1]),
    ]

    # Parameters to run with all value cobination of a parameters but only one of these parameters at a time
    parameters_one_of = [
        ("lr", [None, 0.006, 0.001]),
        ("epochs", [None, 700, 1000]),
        ("batch_size", [None, 6, 12]),
        ("data_augmentation", [None, "randaugment_light_flip", "randaugment_scale_crop_flip"]),
        ("MODEL.PRETRAINED", [None, "False"]),
        ("MODEL.pretrained_on", [None, "Paddle"]),
    ]

    for c, h in parameters_hard:
        job_config += f" {c}={h}"

    command_list = []
    vals = [v[1] for v in parameters_all_of]
    params = [v[0] for v in parameters_all_of]
    combinations = itertools.product(*vals)
    numb = 0
    for combi in combinations:
        numb += 1
        run_conifg_all = ""
        for p_all, c_all in zip(params, combi):
            if c_all is not None:
                run_conifg_all += f"{p_all}={c_all} "

        for p_one, c_one in parameters_one_of:

            for ci_one in c_one:
                run_conifg_one = ""
                if ci_one is not None:
                    run_conifg_one += f"{p_one}={ci_one} "
                final_command = command + f' "{job_config} {run_conifg_all}{run_conifg_one}"'
                command_list.append(final_command)
            # print(final_command)
    command_list = np.unique(command_list)

    for i, com in enumerate(command_list):
        print(com)
        if (i + 1) % 10 == 0:
            print("")

    print(f" --- In Total {len(command_list)} Jobs are configured---")
    """
    Combine Cluster Configuration + Job Configuration
    """
    # command += f' "{job_config}"'
    # print(command)
    # "source ~/.bashrc_sem_seg_2 && python3 training.py model=hrnet_ocr_ms dataset=COMPUTING_2 dataset.fold=0 dataset.random_sampling=0.3 "

    # print(command)
    # "bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=18.0G -L /bin/bash -g /l727r/group_15 -q gpu-lowprio.legacy
