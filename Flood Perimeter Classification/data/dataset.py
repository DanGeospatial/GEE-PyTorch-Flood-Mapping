from cProfile import label

from torch.utils.data import Dataset

class Loader(Dataset):
    def __init__(self, ID_list, labels):
        self.labels = labels
        self.ID_list = ID_list

    def __len__(self):
        # Get total number of samples
        return len(self.ID_list)

    def __getitem__(self, index):
        ID = self.ID_list[index]
    
