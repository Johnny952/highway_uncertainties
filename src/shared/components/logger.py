import wandb

class Logger(object):

    def __init__(self, project_name, group, model_name, run_id, args=None, **kwargs):
        wandb.init(project=project_name, group=group, name=model_name, id=run_id, config=args, **kwargs)

    def get_config(self):
        return wandb.config

    def watch(self, model):
        wandb.watch(model)
    
    def log(self, to_log: dict, step=None):
        wandb.log(to_log, step=step)

    def close(self):
        wandb.finish()

class SimpleLogger(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs["args"]

    def get_config(self):
        return self.args

    def watch(self, model):
        pass

    def log(self, to_log: dict):
        pass

    def close(self):
        pass