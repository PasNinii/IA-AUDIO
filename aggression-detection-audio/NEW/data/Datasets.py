import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, dataset, transform, loading_function):
        self._dataset = dataset
        self._transform = transform
        self._loading_function = loading_function

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        data = self._dataset[index]
        tensor, sr = self._transform.apply(self._loading_function(data["path"]))
        return tensor, sr, data["label"], data["classID"]