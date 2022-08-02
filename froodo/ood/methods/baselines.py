import torch

from .ood_methods import OODMethod
from ...data.datatypes import TaskType

class MaxClassBaseline(OODMethod):
    def __init__(self, hyperparams={}) -> None:
        super().__init__(hyperparams)

    def calculate_ood_score(self, imgs, net, batch=None, task_type = TaskType.SEGMENTATION):
        with torch.no_grad():
            if  task_type == TaskType.SEGMENTATION:
                outputs, *_ = net(imgs.cuda()) 
            else:
                outputs = net(imgs.cuda()) 
        scores = torch.softmax(outputs.data, dim=1)
        m, _ = torch.max(scores, dim=1)
        return m.detach().cpu()
