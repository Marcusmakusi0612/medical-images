import numpy as np
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.evaluation.metric import dice
set_determinism(123)
import os
from light_training.prediction import Predictor

data_dir = "/media/yfl508/My Passport/seg/segmamba-yaozhui/data/fullres/train"
env = "pytorch"
max_epoch = 1000
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:0"
patch_size = [128, 128, 128]

class SegTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        
        self.patch_size = patch_size
        self.augmentation = False

    def convert_labels(self, labels):
        ## 椎间盘和钙化
        result = [
            (labels == 1),  # 区域1
            (labels == 2)  # 区域2
        ]

        return torch.cat(result, dim=1).float()

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
        properties = batch["properties"]
        label = self.convert_labels(label)

        image_affine = image.affine if hasattr(image, 'affine') else None
        return image, label, properties, image_affine

    def define_model_segmamba(self):
        from model_segmamba.segmamba import SegMamba
        model = SegMamba(in_chans=1,
                        out_chans=3,
                        depths=[2,2,2,2],
                        feat_size=[48, 96, 192, 384])
        
        model_path = "/media/yfl508/My Passport/seg/segmamba-yaozhui/logs/segmamba/model/best_model_0.9444.pt"
        new_sd = self.filte_state_dict(torch.load(model_path, map_location="cpu"))
        model.load_state_dict(new_sd)
        model.eval()
        window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.5,
                                        progress=True,
                                        mode="gaussian")

        predictor = Predictor(window_infer=window_infer,
                              mirror_axes=[0,1,2])

        save_path = "./prediction_results/segmamba"
        os.makedirs(save_path, exist_ok=True)

        return model, predictor, save_path

    def validation_step(self, batch):
        image, label, properties, affine = self.get_input(batch)
        original_shape = label.shape[2:]

        model, predictor, save_path = self.define_model_segmamba()
        model_output = predictor.maybe_mirror_and_predict(image, model, device=self.device)

        model_output = torch.nn.functional.interpolate(
            model_output,
            size=original_shape,
            mode='trilinear',
            align_corners=False
        )

        model_output_label = model_output.argmax(dim=1).squeeze(0)  # [D,H,W]

        # One-hot for Dice
        model_output_onehot = self.convert_labels_dim0(model_output_label.unsqueeze(0)).squeeze(0)
        label = label[0]

        dices = []
        for i in range(2):
            output_i = model_output_onehot[i].cpu().numpy()
            label_i = label[i].cpu().numpy()
            dices.append(dice(output_i, label_i))

        print("Dice scores:", dices)

        # 保存 mask 前处理 affine
        model_output_label = model_output_label.float().cpu().numpy()

        # predictor 生成 mask：假设它内部调用的是 nibabel 保存
        model_output_label = predictor.predict_noncrop_probability(
            torch.tensor(model_output_label).unsqueeze(0), properties
        )[0]

        # ✅ 保存时指定 affine 信息
        import nibabel as nib
        nii = nib.Nifti1Image(model_output_label.astype(np.uint8), affine=affine)
        nib.save(nii, os.path.join(save_path, f"{properties['name'][0]}.nii.gz"))

        return 0

    def convert_labels_dim0(self, labels):
        ## 椎间盘和钙化
        result = [
            (labels == 1),  # 区域1
            (labels == 2)  # 区域2
        ]

        return torch.cat(result, dim=0).float()
    

    def filte_state_dict(self, sd):
        if "module" in sd :
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k 
            new_sd[new_k] = v 
        del sd 
        return new_sd
    
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
    
    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(data_dir)

    trainer.validation_single_gpu(test_ds)

    # print(f"result is {v_mean}")


