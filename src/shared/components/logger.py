import wandb

class Logger(object):

    def __init__(self, project_name, group, model_name, run_id, args: None):
        wandb.init(project=project_name, group=group, name=model_name, id=run_id, config=args)

    def get_config(self):
        return wandb.config

    def watch(self, model):
        wandb.watch(model)
    
    def log(self, to_log: dict):
        wandb.log(to_log)