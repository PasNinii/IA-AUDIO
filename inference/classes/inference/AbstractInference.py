import abc

class AbstractInference:
    def __init__(self, model, device, transforms) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.transforms = transforms

    @abc.abstractmethod
    def infer(self, data) -> tuple:
        pass