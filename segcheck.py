import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# 路径
seg_path = "/media/yfl508/My Passport/seg/SegMamba-main/prediction_results/segmamba/BraTS-GLI-00721-001.nii.gz"

# 加载数据
seg_data = np.squeeze(nib.load(seg_path).get_fdata())  # shape: (240, 240, 155)
print("Shape:", seg_data.shape)

# 显示中间切片
slice_index = seg_data.shape[2] // 2
slice_2d = seg_data[:, :, slice_index]

plt.figure(figsize=(6, 6))
plt.imshow(slice_2d.T, cmap='gray', origin='lower')
plt.title(f"Segmentation Slice {slice_index}")
plt.axis('off')
plt.show()
