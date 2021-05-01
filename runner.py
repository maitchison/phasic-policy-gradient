import os
import argparse

DEFAULT_PPG_ARGS = {
    'env_name':     'coinrun',
    'num_envs':     256,
    'epochs':       25,
    'n_aux_epochs': 3,
    'n_pi':         16, # save on the memory...
}

DEFAULT_PPO_ARGS = {
    'env_name':     'coinrun',
    'num_envs':     256,
    'epochs':       25,
    'n_epoch_vf':   3,
    'n_epoch_pi':   3,
    'n_aux_epochs': 0,
    'arch':         'shared',
}

class Job():

    def __init__(self, experiment, job_name, default_args=None, **kwargs):
        self.experiment = experiment
        self.job_name = job_name
        self.args = kwargs
        if default_args is not None:
            for k, v in default_args.items():
                if k not in self.args:
                    self.args[k] = v

    @property
    def experiment_folder(self):
        return f"./run/{self.experiment}"

    @property
    def job_folder(self):
        return f"./run/{self.experiment}/{self.job_name}"

    def run(self, device:str):
        """
        Executes the job
        """

        # 1. copy files to experiment folder for reference (if they are not already there)
        # assuming linux here...
        if not os.path.exists(f"{self.experiment_folder}/train.py"):
            print("Copying source files...")
            os.makedirs(self.experiment_folder, exist_ok=True)
            os.system(f"cp train.py {self.experiment_folder}/")
            os.system(f"cp -r phasic {self.experiment_folder}/")

        # 2. execute the job
        os.chdir(self.experiment_folder)

        workers = device.count(',') + 1

        args = " ".join(f"--{k} {v}" for k, v in self.args.items())

        if workers == 1:
            command_str = f"python train.py '{self.job_name}' --device '{device}' {args}"
        else:
            command_str = f"mpiexec -np {workers} python train.py '{self.job_name}' --device '{device}' {args}"
        print(f"Running: {command_str}")
        os.system(command_str)

    def is_started(self):
        return os.path.exists(self.job_folder)


if __name__ == "__main__":

    JOBS = [
        Job('e_aux', "e_aux=0", DEFAULT_PPG_ARGS, n_aux_epochs=0),
        Job('e_aux', "e_aux=1", DEFAULT_PPG_ARGS, n_aux_epochs=1),
        Job('e_aux', "e_aux=3", DEFAULT_PPG_ARGS, n_aux_epochs=3),
        Job('e_aux', "e_aux=6", DEFAULT_PPG_ARGS, n_aux_epochs=6),
        Job('e_aux', "e_aux=3 (ST)", DEFAULT_PPG_ARGS, n_aux_epochs=3, shuffle_time=True),
        Job('e_aux', "ppo", DEFAULT_PPO_ARGS),
    ]

    parser = argparse.ArgumentParser(description='Run a predefined job')
    parser.add_argument('job_name', type=str)
    parser.add_argument('--device', type=str, default='0')

    args, _ = parser.parse_known_args()

    if args.job_name == "auto":
        todo_jobs = [job for job in JOBS if not job.is_started()]
        if len(todo_jobs) > 0:
            todo_jobs[0].run(device=args.device)
    else:
        raise ValueError("Job name must be 'auto'.")

