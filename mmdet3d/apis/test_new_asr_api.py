import mmcv
import os
from mmdet3d.core.bbox import BaseInstance3DBoxes
import copy


def single_gpu_test_new_asr(data_loader, scattered_result_dir):
    groud_truth = 0
    pre_truth_1 = 0
    pre_truth_2 = 0
    fp_1 = 0
    fp_2 = 0
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
   
    for data_i, data in enumerate(data_loader):
        if data['gt_labels_3d'][0]._data[0][0].shape[0] != 0:
            
            gt_boxes_3d_ori = data['gt_bboxes_3d'][0]._data[0][0]
            output_path = os.path.join(scattered_result_dir, str(data_i)+'.pkl')
            output = mmcv.load(output_path)
            pre_bbox = output[0]['pts_bbox']['boxes_3d']
            pre_score = output[0]['pts_bbox']['scores_3d']
            gt_bbox = copy.deepcopy(gt_boxes_3d_ori)
            ious_3d = BaseInstance3DBoxes.overlaps(gt_bbox, pre_bbox)
            groud_truth += ious_3d.shape[0]
            if ious_3d.shape[1]==0:
                pre_truth_1 += gt_boxes_3d_ori.tensor.shape[0]
                pre_truth_2 += ious_3d.shape[0]
                fp_1 += (pre_score>0.1).sum()
                fp_2 += (pre_score>0.5).sum()
            else:
                for i in range(ious_3d.shape[0]):
                    ious_3d_i, idx = ious_3d[i].topk(k=ious_3d.shape[1], dim=-1)
                    pre_score_truth = pre_score[idx] #[9]
                    if ious_3d_i[0]==0:
                        pre_truth_1 += 1
                        pre_truth_2 += 1
                    else:
                        iou_id = ious_3d_i!=0
                        ### compute asr ### 
                        score_thre_1 = pre_score_truth[ious_3d_i!=0]<0.1
                        score_thre_2 = pre_score_truth[ious_3d_i!=0]<0.5
                        if score_thre_1.all():
                            pre_truth_1 += 1
                        if score_thre_2.all() >0:
                            pre_truth_2 += 1
                ### 计算大于0.1的bbox以及大于0.5的bbox的个数 ######
                num_1 = 0
                num_2 = 0
                pre_score_1 = pre_score[pre_score>0.1]
                pre_score_2 = pre_score[pre_score>0.5]
                pre_bbox_1 = pre_bbox[pre_score>0.1]
                pre_bbox_2 = pre_bbox[pre_score>0.5]
                if pre_bbox_1.tensor.shape[0]==0:
                    num_1 += 0
                else:
                    ious_3d_1 = BaseInstance3DBoxes.overlaps(pre_bbox_1, gt_bbox)
                    for i in range(ious_3d_1.shape[0]):
                        ious_3d_i, idx = ious_3d_1[i].topk(k=ious_3d_1.shape[1], dim=-1)
                        if ious_3d_i[0]==0:
                            num_1 += 1
                        elif (ious_3d_i[0]!=0)>0:
                            num_1 += 0
                    
                if pre_bbox_2.tensor.shape[0]==0:
                    num_2 += 0
                else:
                    ious_3d_2 = BaseInstance3DBoxes.overlaps(pre_bbox_2, gt_bbox)
                    for i in range(ious_3d_2.shape[0]):
                        ious_3d_i, idx = ious_3d_2[i].topk(k=ious_3d_2.shape[1], dim=-1)
                        if ious_3d_i[0]==0:
                            num_2 += 1
                        elif (ious_3d_i[0]!=0)>0:
                            num_2 += 0
                fp_1 += num_1
                fp_2 += num_2
        prog_bar.update()
    asr_1 = pre_truth_1 / groud_truth
    asr_2 = pre_truth_2 / groud_truth
    return asr_1, asr_2, fp_1, fp_2