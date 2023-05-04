from src.callbacks.loss.loss_field import compute_loss
from src.callbacks.process.process_field import process_data
from src.callbacks.vis.visualize_field import visualize_all
from src.models.field_sf.model import FieldSF
from src.models.generic.wrapper import GenericWrapper


class FieldSFWrapper(GenericWrapper):
    def __init__(self, args):
        super().__init__(args)
        self.model = FieldSF("resnet50", args.focal_length, args.img_res)
        self.process_fn = process_data
        self.loss_fn = compute_loss
        self.metric_dict = ["avg_err_field"]

        self.vis_fns = [visualize_all]
        self.num_vis_train = 1
        self.num_vis_val = 1

    def inference(self, inputs, meta_info):
        return super().inference_field(inputs, meta_info)
