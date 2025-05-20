from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor
import numpy as np
import pickle
import json

data_filename = ["t.nii.gz"]



base_dir = r"/media/yfl508/My Passport/seg/segmamba-yaozhui/data/all-yaozhui rawdata/"
image_dir = "Yao zhui Validation Data"


def process_train():
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir,
                                             image_dir=image_dir,
                                             data_filenames=data_filename,

                                             )

    out_spacing = [1.0, 1.0, 1.0]
    output_dir = r"/media/yfl508/My Passport/seg/segmamba-yaozhui/data/fullres/infer"

    preprocessor.run(output_spacing=out_spacing,
                     output_dir=output_dir,
                     all_labels=[]
                     )


def plan():
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir,
                                             image_dir=image_dir,
                                             data_filenames=data_filename,

                                             )

    preprocessor.run_plan()


if __name__ == "__main__":
    plan()
    process_train()

