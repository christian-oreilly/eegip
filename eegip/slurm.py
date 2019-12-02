import sys
import os
import subprocess
from jinja2 import Template


def get_dep_key(job_key, dep_dict):
    dep_dict_key = list(dep_dict.keys())[0]
    return ((key, dict(job_key)[key]) for key, val in dep_dict_key)


def save_and_launch(job_name, file_name_log, file_name_slurm, program_call,
                    eegip_config, config_task_root, dep_after=None):

    with open(os.path.join(os.path.dirname(__file__), "scripts", "slurm_template.jinja"), "r") as file_jinja:
        jinja_template = file_jinja.read()

    slurm_script = Template(jinja_template).render(job_name=job_name,
                                                   email=eegip_config["slurm"]["email"],
                                                   send_emails=eegip_config["slurm"]["send_emails"],
                                                   file_name_log=file_name_log,
                                                   ntask=config_task_root["ntask"],
                                                   time=config_task_root["time"],
                                                   mem_per_cpu=config_task_root["mem_per_cpu"],
                                                   account=eegip_config["slurm"]["account"],
                                                   venv_path=eegip_config["paths"]["venv_path"],
                                                   program_call=program_call)

    print("Saving and launching ", file_name_slurm)
    if not os.path.exists(eegip_config["paths"]["log_dir"]):
        os.makedirs(eegip_config["paths"]["log_dir"])
    if not os.path.exists(eegip_config["paths"]["slurm_dir"]):
        os.makedirs(eegip_config["paths"]["slurm_dir"])
    with open(file_name_slurm, 'w') as file_slurm:
        file_slurm.write(slurm_script)

    args = ["sbatch"]
    if dep_after is not None:
        args.append("--dependency=afterok:" + str(dep_after))
    args.append(file_name_slurm)
    print(" ".join(args))
    res = subprocess.check_output(args).strip()
    print(res.decode(), file=sys.stdout)
    if not res.startswith(b"Submitted batch"):
        return None
    return int(res.split()[-1])
