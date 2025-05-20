import numpy as np
from light_training.dataloading.dataset import get_test_loader_from_test
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.prediction import Predictor
import os

set_determinism(123)

# Directories and settings
data_dir = "/media/yfl508/My Passport/seg/segmamba-yaozhui/data/fullres/infer"  # Test data (no 'seg' files)
env = "pytorch"
max_epoch = 1000
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:0"
patch_size = [128, 128, 128]


class SegTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port,
                         training_script)

        self.patch_size = patch_size
        self.augmentation = False

    def get_input(self, batch):
        # Only extract image, no label (seg)
        image = batch["data"]
        properties = batch["properties"]

        return image, properties

    def define_model_segmamba(self):
        from model_segmamba.segmamba import SegMamba
        model = SegMamba(in_chans=1,
                         out_chans=3,
                         depths=[2, 2, 2, 2],
                         feat_size=[48, 96, 192, 384])

        model_path = "/media/yfl508/My Passport/seg/segmamba-yaozhui/logs/segmamba/model/best_model_0.9444.pt"
        new_sd = self.filte_state_dict(torch.load(model_path, map_location="cpu"))
        model.load_state_dict(new_sd)
        model.eval()

        # Sliding window for inference
        window_infer = SlidingWindowInferer(roi_size=patch_size,
                                            sw_batch_size=2,
                                            overlap=0.5,
                                            progress=True,
                                            mode="gaussian")

        # Initialize the predictor for inference
        predictor = Predictor(window_infer=window_infer,
                              mirror_axes=[0, 1, 2])

        save_path = "./prediction_results/segmamba"
        os.makedirs(save_path, exist_ok=True)

        return model, predictor, save_path

    def inference_step(self, batch):
        import nibabel as nib

        # Get image and properties (no label here)
        image, properties = self.get_input(batch)
        model, predictor, save_path = self.define_model_segmamba()

        # Predict model output
        model_output = predictor.maybe_mirror_and_predict(image, model, device=device)

        # Get raw prediction (probability)
        model_output = predictor.predict_raw_probability(model_output, properties=properties)

        # Argmax to get class label map: shape = (H, W, D)
        model_output = model_output.argmax(dim=0).cpu().numpy()

        # Restore non-cropped shape
        model_output_noncrop = predictor.predict_noncrop_probability(model_output[None], properties)[0]

        # Check and transpose if shape is (240, 155, 240)
        if model_output_noncrop.shape == (240, 155, 240):
            model_output_noncrop = np.transpose(model_output_noncrop, (0, 2, 1))  # -> (240, 240, 155)

        # --- Find reference image (t1c.nii.gz) ---
        case_name = properties['name'][0]  # e.g., 'BraTS-GLI-00001-000'
        ref_image_path = os.path.join(
            "/media/yfl508/My Passport/seg/segmamba-yaozhui/data/all-yaozhui rawdata/Yao zhui Validation Data",
            case_name,
            "t.nii.gz"
        )

        # --- Load the reference image ---
        # ref_img = nib.load(ref_image_path)

        # --- Save prediction aligned to reference image ---
        predictor.save_to_nii(model_output_noncrop,
                              raw_spacing=[1, 1, 1],
                              reference_nifti_path=ref_image_path,
                              case_name=case_name,
                              save_dir=save_path)

        return 0

        # 保存
        # predictor.save_to_nii(model_output_noncrop,
        #                       raw_spacing=[1, 1, 1],
        #                       case_name=properties['name'][0],
        #                       save_dir=save_path)


    def convert_labels_dim0(self, labels):
        ## 椎间盘和钙化
        result = [
            (labels == 1),  # 区域1
            (labels == 2)  # 区域2
        ]

        return torch.cat(result, dim=0).float()

    def filte_state_dict(self, sd):
        if "module" in sd:
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k
            new_sd[new_k] = v
        del sd
        return new_sd

    def validation_step(self, batch):
        return self.inference_step(batch)

if __name__ == "__main__":
    trainer = SegTrainer(env_type=env,
                           max_epochs=max_epoch,
                           batch_size=batch_size,
                           device=device,
                           logdir="",
                           val_every=val_every,
                           num_gpus=num_gpus,
                           master_port=17751,
                           training_script=__file__)

    # Test data only (no labels)
    test_ds = get_test_loader_from_test(data_dir)  # Only get the test data

    trainer.validation_single_gpu(test_ds)  # Perform inference and save segmentation results

