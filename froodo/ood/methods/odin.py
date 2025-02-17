import torch
from torch.autograd import Variable

from .ood_methods import OODMethod
from ...data.datatypes import TaskType


class ODIN(OODMethod):
    def __init__(self, hyperparams={"temperature": 2, "noise": 0.004}) -> None:
        super().__init__()
        self.temperature = hyperparams["temperature"]
        self.noise = hyperparams["noise"]
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_params(self, dict=False):
        if dict:
            return {"temperature": self.temperature, "noise": self.noise}
        return f"temp: {self.temperature}, noise: {self.noise}"

    def _temp_and_pertubate(self, imgs, net, task_type = TaskType.SEGMENTATION):
        inputs = Variable(imgs.cuda(), requires_grad=True)
        if  task_type == TaskType.SEGMENTATION:
            outputs, *_ = net(inputs)
        else:
            outputs = net(inputs)

        max = torch.max(outputs.data, axis=1)[0]
        scores = torch.softmax(outputs.data - max.unsqueeze(1), dim=1)

        labels = torch.argmax(scores, axis=1)
        loss = self.criterion(outputs / self.temperature, labels)
        loss.backward()

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        tempInputs = torch.add(inputs.data, gradient, alpha=-self.noise)
        if  task_type == TaskType.SEGMENTATION:
            outputs, *_ = net(tempInputs)
        else:
            outputs = net(tempInputs)
        outputs = outputs / self.temperature
        return outputs

    def calculate_ood_score(self, imgs, net, batch=None, task_type = TaskType.SEGMENTATION):
        outputs = self._temp_and_pertubate(imgs, net,task_type)
        outputs = outputs - torch.max(outputs, axis=1)[0].unsqueeze(1)
        scores = torch.softmax(outputs.data, dim=1)
        m, _ = torch.max(scores, axis=1)
        return m.detach().cpu()
