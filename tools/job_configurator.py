import itertools
import numpy as np


class Job_Configurator:
    def __init__(self, script, bash, gmem="10.0G", num_gpus=1, queue="gpu-lowprio", job_group=None):
        self.command = (
            f"bsub -gpu num={num_gpus}:j_exclusive=yes:mode=exclusive_process:gmem={gmem} -L"
            f" /bin/bash -q {queue}"
        )

        if job_group is not None:
            self.command += f" -g {job_group}"

        self.job_config = f"source ~/{bash} && python3 {script} "

        self.parameters_hard = []
        self.parameters_grid_search = []
        self.parameter_random_search = []

    def add_hard_set_parameter(self, parameter, value):
        self.parameters_hard.append((parameter, value))

    def add_grid_search_parameter(self, parameter, values):
        self.parameters_grid_search.append((parameter, values))

    def add_random_search_parameter(self, parameter, values, stype):
        self.parameter_random_search.append((parameter, values, stype))

    def run_grid_search(self):
        run_configs = []

        vals = [v[1] for v in self.parameters_grid_search]
        params = [v[0] for v in self.parameters_grid_search]
        combinations = itertools.product(*vals)

        for combi in combinations:
            run_config = ""
            for p_all, c_all in zip(params, combi):
                if c_all is not None:
                    run_config += f"{p_all}={c_all} "
            run_configs.append(run_config)

        run_configs = np.unique(run_configs)

        return run_configs

    def run_random_search(self):
        config = ""
        for parameter, value_range, stype in self.parameter_random_search:
            if stype == "list":
                val = np.random.choice(value_range, 1)[0]
            elif stype == "int":
                val = np.random.randint(value_range[0], value_range[1] + 1, 1)[0]
            elif stype == "float":
                val = np.random.uniform(value_range[0], value_range[1], size=1)[0]
            config += f"{parameter}={val:.5f} " if stype == "float" else f"{parameter}={val} "
        return config

    def run(self, num_random_searchs=5):
        for c, h in self.parameters_hard:
            self.job_config += f"{c}={h} "

        run_conifgs = self.run_grid_search()
        runs = []

        for run_conifg in run_conifgs:
            if num_random_searchs == 0 or self.parameter_random_search == []:
                runs.append(self.command + self.job_config + run_conifg)
            for i in range(0, num_random_searchs):
                # runs.append(self.command + self.job_config + run_conifg + self.run_random_search())
                runs.append(
                    self.command + f' "{self.job_config} {run_conifg}{self.run_random_search()}"'
                )
        runs = np.unique(runs)

        line_break = 8
        for i, com in enumerate(runs):
            print(com)
            if (i + 1) % line_break == 0:
                print("")

        print(f" --- In Total {len(runs)} Jobs are configured---")


if __name__ == "__main__":
    """
    Cluster and Job Configuration
    """
    script = "training.py"
    bash = ".bashrc_sem_seg_2"
    queue = "gpu-lowprio"
    gmem = "18.0G"
    num_gpus = 1
    job_group = "/l727r/group_20"

    jobconf = Job_Configurator(script, bash, gmem, num_gpus, queue, job_group)

    """
    Set Hard Parameters - Set in each run
    """
    jobconf.add_hard_set_parameter("dataset", "Solar_Hydrogen")
    jobconf.add_hard_set_parameter("environment", "cluster")
    jobconf.add_hard_set_parameter("pl_trainer.enable_checkpointing", "False")
    jobconf.add_hard_set_parameter("augmentation", "randaugment_nonorm_flip")

    """
    Set Grid Search Parameters - Test each possible combination
    """
    # jobconf.add_grid_search_parameter(
    #     "augmentation",
    #     [
    #         "randaugment_nonorm_flip",
    #         "randaugment_light_nonorm_flip",
    #     ],
    # )
    jobconf.add_grid_search_parameter("model.version", ["v1", "v2"])
    jobconf.add_grid_search_parameter("model", ["Mask_RCNN", "Mask_RCNN_RMI"])

    """
    Set Random Search Parameters - Select a random value in given range for each parameter
    """
    jobconf.add_random_search_parameter("lr", [0.0015, 0.005], "float")
    jobconf.add_random_search_parameter("epochs", [50, 300], "int")
    jobconf.add_random_search_parameter("batch_size", [2, 8], "int")
    jobconf.add_random_search_parameter("AUGMENTATIONS.N", [1, 5], "int"),
    jobconf.add_random_search_parameter("AUGMENTATIONS.M", [1, 5], "int"),
    # jobconf.add_random_search_parameter(
    #     "++lr_scheduler.scheduler.warmstart_iters", [0.0075, 0.0175], "float"
    # ),
    # jobconf.add_random_search_parameter("lr_scheduler", ["polynomial_warmup", "polynomial"], "list")

    jobconf.run(num_random_searchs=20)
