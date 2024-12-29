#🌐 IoU-S Attack
Official implementation of the paper ''Efficient Adversarial Attack Strategy Against 3D Object Detection in Autonomous Driving Systems''.(Note: This is a reference to the implementation of IoU-S Attack on mmdetection3d.)
## Installation

- Clone mmdet3d repo
    ```bash
    git clone https://github.com/open-mmlab/mmdetection3d.git
    git checkout -v v0.17.2
    ```
    Install the necessary dependencies according to [Installation](https://mmdetection3d.readthedocs.io/en/latest/get_started.html)
## 🌟 Files in the folder
- `IoU-S-Attack/`
  - `config/`: The config we changed for centrepoint..
  - `mmdet3d/`
    - `test_iou_api.py`: Implementation of IoU-S Attack.
    - `test_new_asr.py/`: Implementation of evaluation measures.
  - `tools/`
    - `test_iou.py`: Running adversarial attacks.
    - `test_new_asr.py`: Showing evaluation metrics.
## 💾 Some explanations
1. Put the above files into the corresponding folder in mmdetection3d.
   
2. For the problem that voxelisation is not differentiable, see our other paper solution [Bev-Robust](https://github.com/zzj403/BEV_Robust.git).

3. For chamfer distances we use [chamferdist](https://github.com/OpenDriveLab/ViDAR/tree/936dbf7e010189b68b83b4b61568cfd0fa23e655/third_lib/chamfer_dist/chamferdist).

4. For the implementation of other 3D object detection methods, please refer to our examples.

5. In the actual code deployment process, there are detach, with torch.no_grad() in some methods, these are the cause of the gradient disconnection, please debug carefully for details, or refer to our another work [Bev-Robust](https://github.com/zzj403/BEV_Robust.git).

6. Although we do not provide a multi-card distributed run script, our code supports multi-card serial and can be run directly on multiple cards.

## 📘 Explanation of key parameters
1. --scattered_result_dir_adv：Store the results after adversarial attacks as a pkl file

2. lidar_save_path(test_iou_api.py)：Store adversarial point clouds allowing for self-defined storage paths

## 📝 Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{chen2024efficient,
  title={Efficient Adversarial Attack Strategy Against 3D Object Detection in Autonomous Driving Systems},
  author={Chen, Hai and Yan, Huanqian and Yang, Xiao and Su, Hang and Zhao, Shu and Qian, Fulan},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2024},
  publisher={IEEE}
}
```

