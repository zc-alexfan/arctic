from src.datasets.arctic_dataset import ArcticDataset


class ArcticDatasetEval(ArcticDataset):
    def getitem(self, imgname, load_rgb=True):
        return self.getitem_eval(imgname, load_rgb=load_rgb)
