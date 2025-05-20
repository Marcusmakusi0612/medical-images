import os
import shutil


def standardize_filenames(root_dir):
    """
    将每个case文件夹下的文件重命名为标准名称：
    {case}.nii.gz → t.nii.gz
    {case}seg.nii.gz → seg.nii.gz
    """
    for case_dir in os.listdir(root_dir):
        case_path = os.path.join(root_dir, case_dir)

        # 确保是目录
        if not os.path.isdir(case_path):
            continue

        # 查找原始文件
        orig_img = os.path.join(case_path, f"{case_dir}.nii.gz")
        orig_seg = os.path.join(case_path, f"{case_dir}seg.nii.gz")

        # 新文件名
        new_img = os.path.join(case_path, "t.nii.gz")
        new_seg = os.path.join(case_path, "seg.nii.gz")

        # 重命名图像文件
        if os.path.exists(orig_img):
            os.rename(orig_img, new_img)
            print(f"Renamed: {orig_img} → {new_img}")
        else:
            print(f"Warning: Missing image file in {case_dir}")

        # 重命名分割文件
        if os.path.exists(orig_seg):
            os.rename(orig_seg, new_seg)
            print(f"Renamed: {orig_seg} → {new_seg}")
        else:
            print(f"Warning: Missing segmentation file in {case_dir}")


if __name__ == '__main__':
    target_dir = input("请输入包含case文件夹的根目录路径: ").strip()
    if os.path.isdir(target_dir):
        standardize_filenames(target_dir)
        print("文件名标准化完成！")
    else:
        print("错误：指定的路径不是目录或不存在")