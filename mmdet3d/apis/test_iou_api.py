# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp
import os
import mmcv
import torch
import pickle
import numpy as np
import copy
import torch.optim as optim
from mmdet3d.core.bbox import BaseInstance3DBoxes
from chamferdist import ChamferDistance
dist_func = ChamferDistance()


def save_pickle(data, file_name):
    f = open(file_name, "wb")
    pickle.dump(data, f)
    f.close()


def single_gpu_test_iou(model, data_loader, scattered_result_dir_adv, attack_name,
                        attack_lr, steps, model_name, sub_loss, num_add, num_drop, k_drop_round):
    model.eval()
    results = []
    total_num = sum(p.numel() for p in model.parameters())
    print('the parameters of model: %.5f' %(total_num))
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    scattered_result_dir_adv = scattered_result_dir_adv + '_' + str(model_name) + '_' + str(attack_name) + '_' + str(sub_loss)
    os.makedirs(scattered_result_dir_adv, exist_ok=True)
    
    for data_i, data in enumerate(data_loader):
        
        #多进程进行adv
        scattered_result_path_adv = os.path.join(scattered_result_dir_adv, str(data_i)+'.pkl')
        if os.path.exists(scattered_result_path_adv):
            print(scattered_result_path_adv, 'exists! pass!')
            continue
        else:
            mmcv.dump(' ', scattered_result_path_adv)
        
        #transfor data device to cuda
        device = model.src_device_obj
        points_ori = data['points'][0]._data[0][0].to(device)
        gt_boxes_3d_ori = data['gt_bboxes_3d'][0]._data[0][0]
        
         ####definite lidar save path
        lidar_save_path = 'adv_outputs/%s_adv_%s_%s' %(model_name, attack_name, sub_loss)
        if not os.path.exists(lidar_save_path):
            os.makedirs(lidar_save_path)
        
        lidar_path = data['img_metas'][0]._data[0][0]['pts_filename']
        
        lidar_trans_path = lidar_path.replace('./data/nuscenes/samples/LIDAR_TOP', lidar_save_path)

        lidar_trans_path = lidar_trans_path.replace('.pcd.bin', '.pkl')
        
        #走训练的时候需要剥开外面的壳
        data['points'] = data['points'][0]
        data['img_metas'] = data['img_metas'][0]
        data['gt_bboxes_3d'] = data['gt_bboxes_3d'][0]
        data['gt_labels_3d'] = data['gt_labels_3d'][0]
        # 防止出现no_gt
        data['img_metas']._data[0][0]['return_gpu'] = True
        if data['gt_labels_3d']._data[0][0].shape[0] != 0:
            if attack_name=='iou_per':
                #random start
                delta = torch.rand_like(points_ori) * 1e-2
                adv_point = points_ori + delta #only xyz
                # 点云不用clamp, 但5维点云的后2维不能改
                if model_name=='centerpoint':
                    adv_point[:,-2:] = points_ori[:,-2:]
                else:
                    adv_point[:,-1:] = points_ori[:,-1:]

                adv_point = adv_point.clone().detach().requires_grad_()
                opt = optim.Adam([adv_point], lr=attack_lr, weight_decay=0.)
                
                o_bestdist = np.array([1e10])
                o_bestscore = np.array([1e10])
                o_bestattack = np.zeros((1, adv_point.shape[0], adv_point.shape[1]))
                #测试的时候需要把外面一层还回来
                data['points'] = [data['points']]
                data['img_metas'] = [data['img_metas']]
                data['gt_bboxes_3d'] = [data['gt_bboxes_3d']]
                data['gt_labels_3d'] = [data['gt_labels_3d']]
                for step in range(steps):  
                    
                    data['points'][0]._data[0][0] = adv_point.float()                 
                    result = model(return_loss=False, rescale=True, **data)
                    pre_bbox = result[0]['pts_bbox']['boxes_3d']
                    pre_score = result[0]['pts_bbox']['scores_3d']
                    
                    gt_bbox = copy.deepcopy(gt_boxes_3d_ori)
                    ious_3d = BaseInstance3DBoxes.overlaps(gt_bbox.to(device), pre_bbox)
                    ious_3d_sorted, idx = ious_3d.topk(k=pre_bbox.tensor.shape[0], dim=-1) #[9, 200]
                    
                    if ious_3d_sorted.shape[1]==0:
                        adv_point = adv_point
                    else:
                    
                        if idx.shape[1]==0:
                            o_bestattack[0] = adv_point.detach().cpu().numpy()  
                        else:
                            loss = 0
                            for j in range(idx.shape[-1]):
                                pre_score_truth = pre_score[idx[:,j]]#[9]
                                ious_3d = ious_3d_sorted[:, j] #[9]   
                                if sub_loss=='iou':
                                    loss_ = -(torch.log(1-ious_3d+1e-8))
                                elif sub_loss=='score':
                                    loss_ = -(torch.log(1 - pre_score_truth+1e-8))
                                elif sub_loss=='all':
                                    loss_ = -(torch.log(1-pre_score_truth+1e-8) + torch.log(1 - ious_3d+1e-8))
                                loss += loss_
                            #compute dist loss
                            dist1 = dist_func(adv_point[:, :3][None,:,:], points_ori[:, :3][None,:,:])
                            dist2 = dist_func(points_ori[:, :3][None,:,:], adv_point[:, :3][None,:,:])
                            dist_loss = dist1 + dist2
                            
                            dist_loss = dist_loss + torch.sqrt(torch.sum((adv_point[:, :3] - points_ori[:, :3]) ** 2, dim=[0, 1]) + 1e-4)
                            loss_all = loss.sum() + dist_loss
                            opt.zero_grad()
                            loss_all.backward()
                            opt.step() 
                            if model_name=='centerpoint':
                                adv_point[:,-2:].data = points_ori[:,-2:].data
                            else:
                                adv_point[:,-1:].data = points_ori[:,-1:].data
                            print('iteration {}, adv_loss: {:.4f}, dist_loss: {:.10f}'.format(step, loss.sum(), dist_loss.item()))
                            # record values!
                            dist_val = torch.sqrt(torch.sum((adv_point[:, :3] - points_ori[:, :3]) ** 2, dim=[0, 1]))[None].detach().cpu().numpy() # [B]
                            if step==0:
                                dist_val = dist_val + 10.0
                            adv_loss_val = loss_all[None].cpu().detach().numpy()
                            input_val = adv_point[None,:,:].detach().cpu().numpy()  # [K, 3]
                            # update
                            for e, (dist, loss_1, ii) in enumerate(zip(dist_val, adv_loss_val, input_val)):
                                if dist < o_bestdist[e] and loss_1 < o_bestscore[e]:
                                    o_bestdist[e] = dist
                                    o_bestscore[e] = loss_1
                                    o_bestattack[e] = ii   
            
            elif attack_name=='iou_drop':
                num_rounds = int(np.ceil(float(num_drop) / float(k_drop_round)))
                adv_point = points_ori
                #测试的时候需要把外面一层还回来
                data['points'] = [data['points']]
                data['img_metas'] = [data['img_metas']]
                data['gt_bboxes_3d'] = [data['gt_bboxes_3d']]
                data['gt_labels_3d'] = [data['gt_labels_3d']]
                for i in range(num_rounds):
                    input_pc = adv_point.clone().detach().requires_grad_()
                    K = input_pc.shape[0]
                    # number of points to drop in this round
                    k_round = min(k_drop_round, num_drop - i * k_drop_round) #2
                    data['points'][0]._data[0][0] = input_pc 
                    result = model(return_loss=False, rescale=True, **data)
                    pre_bbox = result[0]['pts_bbox']['boxes_3d']
                    pre_score = result[0]['pts_bbox']['scores_3d']
                    
                    gt_bbox = copy.deepcopy(gt_boxes_3d_ori)
                    ious_3d = BaseInstance3DBoxes.overlaps(gt_bbox.to(device), pre_bbox)
                    ious_3d_sorted, idx = ious_3d.topk(k=pre_bbox.tensor.shape[0], dim=-1) #[1, 200]
                    if ious_3d_sorted.shape[1]==0:
                        adv_point = adv_point
                    else:
                        loss = 0
                        for j in range(idx.shape[-1]):
                            pre_score_truth = pre_score[idx[:,j]] #[9]
                            ious_3d = ious_3d_sorted[:, j] #[9]   
                            if sub_loss=='iou':
                                loss_ = (torch.log(1-ious_3d+1e-8))
                            elif sub_loss=='score':
                                loss_ = (torch.log(1 - pre_score_truth+1e-8))
                            elif sub_loss=='all':
                                loss_ = (torch.log(1-pre_score_truth+1e-8) + torch.log(1 - ious_3d))
                            loss = loss + loss_
                        loss_all = loss.sum()
                        loss_all.backward()
                        
                        grad = input_pc.grad.data  #  [K, 5]
                        grad = torch.sum(grad ** 2, dim=1)  # [K]
                        #从小到大排序
                        _, idx = (-grad).topk(k=K-k_round, dim=-1) #[num]
                        adv_point = input_pc[idx]   
            
            elif attack_name=='iou_add':
                #get critical point, points.shape (k, 5)
                input_pc = points_ori.clone().detach().requires_grad_()
                data['points']._data[0][0] = input_pc
                #测试的时候需要把外面一层还回来
                data['points'] = [data['points']]
                data['img_metas'] = [data['img_metas']]
                data['gt_bboxes_3d'] = [data['gt_bboxes_3d']]
                data['gt_labels_3d'] = [data['gt_labels_3d']]
                result = model(return_loss=False, rescale=True, **data)
                pre_bbox = result[0]['pts_bbox']['boxes_3d']
                pre_score = result[0]['pts_bbox']['scores_3d']
                
                gt_bbox = copy.deepcopy(gt_boxes_3d_ori)
                ious_3d = BaseInstance3DBoxes.overlaps(gt_bbox.to(device), pre_bbox)
                ious_3d_sorted, idx = ious_3d.topk(k=pre_bbox.tensor.shape[0], dim=-1) #[1, 200]
                loss = 0
                for j in range(idx.shape[-1]):
                    pre_score_truth = pre_score[idx[:,j]] #[9]
                    ious_3d = ious_3d_sorted[:, j] #[9]   
                    if sub_loss=='iou':
                        loss_ = (torch.log(1-ious_3d+1e-8))
                    elif sub_loss=='score':
                        loss_ = (torch.log(1 - pre_score_truth+1e-8))
                    elif sub_loss=='all':
                        loss_ = (torch.log(1-pre_score_truth+1e-8) + torch.log(1 - ious_3d))
                    loss = loss + loss_
                loss_all = loss.sum()
                loss_all.backward()
                with torch.no_grad():
                    grad = input_pc.grad.data  #  [K, 5]
                    grad = torch.sum(grad ** 2, dim=1)  #[K]
                    _, idx = grad.topk(k=num_add, dim=-1)
                    critical_points = points_ori[idx]
                
                #init critical points with random start
                delta = torch.randn_like(critical_points) * 1e-7
                adv_point = critical_points + delta
                # 点云不用clamp, 但5维点云的后1维不能改
                if model_name=='centerpoint':
                    adv_point[:,-2:] = critical_points[:,-2:]
                else:
                    adv_point[:,-1:] = critical_points[:,-1:]
                #add attack
                adv_point = adv_point.clone().detach().requires_grad_()
                opt = optim.Adam([adv_point], lr=attack_lr, weight_decay=0.)
                
                for step in range(steps):  
                    cat_data = torch.cat([points_ori, adv_point], dim=0)
                    # cat_data = cat_data.clone().detach().requires_grad_()
                    data['points'][0]._data[0][0] = cat_data
                    result = model(return_loss=False, rescale=True, **data)
                    loss = 0
                    pre_bbox = result[0]['pts_bbox']['boxes_3d']
                    pre_score = result[0]['pts_bbox']['scores_3d']
                    
                    gt_bbox = copy.deepcopy(gt_boxes_3d_ori)
                    ious_3d = BaseInstance3DBoxes.overlaps(gt_bbox.to(device), pre_bbox)
                    ious_3d_sorted, idx = ious_3d.topk(k=pre_bbox.tensor.shape[0], dim=-1) #[9, 200]
                    
                    if ious_3d_sorted.shape[1]==0:
                        adv_point = adv_point
                    else:
                    
                        if idx.shape[1]==0:
                            adv_point = adv_point.clone().detach() 
                        else:
                            loss = 0
                            for j in range(idx.shape[-1]):
                                pre_score_truth = pre_score[idx[:,j]] #[9]
                                ious_3d = ious_3d_sorted[:, j] #[9]   
                                if sub_loss=='iou':
                                    loss_ = -(torch.log(1-ious_3d+1e-8))
                                elif sub_loss=='score':
                                    loss_ = -(torch.log(1 - pre_score_truth+1e-8))
                                elif sub_loss=='all':
                                    loss_ = -(torch.log(1-pre_score_truth+1e-8) + torch.log(1 - ious_3d+1e-8))
                                loss += loss_
                            #compute dist loss
                            dist1 = dist_func(adv_point[:, :3][None,:,:], points_ori[:, :3][None,:,:])
                            # dist2 = dist_func(points_ori[:, :3][None,:,:], adv_point[:, :3][None,:,:])
                            dist_loss = dist1
                            loss_all = dist_loss + loss.sum()
                            opt.zero_grad()
                            loss_all.backward()
                            opt.step() 
                            if model_name=='centerpoint':
                                adv_point[:,-2:].data = critical_points[:,-2:].data
                            else:
                                adv_point[:,-1:].data = critical_points[:,-1:].data
                            adv_point.detach().requires_grad_()
                            print('iteration {}, adv_loss: {:.4f}, dist_loss: {:.10f}'.format(step, loss.sum(), dist_loss.item()))
            
            if attack_name=='iou_per':
                adv_point = torch.from_numpy(o_bestattack[0]) 

            elif attack_name=='iou_drop':
                adv_point = adv_point
 
            elif attack_name=='iou_add':
                adv_point = torch.cat([points_ori, adv_point], dim=0) 
               
        else:
            data['points'] = [data['points']]
            data['img_metas'] = [data['img_metas']]
            data['gt_bboxes_3d'] = [data['gt_bboxes_3d']]
            data['gt_labels_3d'] = [data['gt_labels_3d']]
            adv_point = points_ori
        ####save point cloud
        save_pickle(adv_point, lidar_trans_path)
        
        data['points'][0]._data[0][0] = adv_point.float()
        data['img_metas'][0]._data[0][0]['return_gpu'] = False
        #测试的时候需要把外面一层还回来
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        results.extend(result)
        mmcv.dump(result, scattered_result_path_adv)
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
