# Taming Self-Training for Open-Vocabulary Object Detection（CVPR2024）
[arXiv](https://arxiv.org/abs/2308.06412)


## Download pretrained weights
- Download various [RegionCLIP's pretrained weights](https://drive.google.com/drive/folders/1hzrJBvcCrahoRcqJRqzkIGFO_HUSJIii). Check [here](https://github.com/microsoft/RegionCLIP/blob/main/docs/MODEL_ZOO.md#model-downloading) for more details.
Create a new folder `pretrained_ckpt` to put those weights. In this repository, `regionclip`, `concept_emb` and `rpn` will be used.

- Download [our pretrained weights](https://drive.google.com/drive/u/1/folders/1TAr7nZSvpB6nCZCC6nXBw6xgmMmlL0X9) and put them in corresponding folders in `pretrained_ckpt`. 
Our pretrained weights includes:
    - `r50_3x_pre_RegCLIP_cocoRPN_2`: RPN weights pretrained only with COCO Base categories. This is used for experiments on COCO to avoid potential data leakage.
    - `concept_emb`: Complementary to RegionCLIP's `concept_emb`.

## Evaluation with released weights

### Results on COCO-OVD
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Configs</th>
<th valign="bottom">Novel AP</th>
<th valign="bottom">Base AP</th>
<th valign="bottom">Overall AP</th>
<!-- TABLE BODY -->
<!-- ROW: with LSJ -->
 <tr><td align="left"><a href="./sas_det/configs/regionclip/COCO-InstanceSegmentation/customized/CLIP_fast_rcnn_R_50_C4_ovd_PLs.yaml">w/o SAF head</a></td>
<td align="center">31.4</td>
<td align="center">55.7</td>
<td align="center">49.4</td>
</tr>
<!-- ROW: with out LSJ -->
 <tr><td align="left"><a href="./sas_det/configs/ovd_coco_R50_C4_ensemble_PLs.yaml">with SAF head</a></td>
<td align="center">37.4</td>
<td align="center">58.5</td>
<td align="center">53.0</td>
</tr>
</tbody></table>


<details>
<summary>
training with command as the script,
</summary>
  
```bash
python3 ./test_net.py \
    --num-gpus 8 \
    --config-file ./sas_det/configs/ovd_coco_R50_C4_ensemble_PLs.yaml \
    MODEL.WEIGHTS ./pretrained_ckpt/sas_det/regionclip_pretrained-cc_r50.pth \
    MODEL.CLIP.OFFLINE_RPN_CONFIG ./sas_det/configs/regionclip/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
    MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
    MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_48_base_cls_emb.pth \
    MODEL.CLIP.CONCEPT_POOL_EMB ./pretrained_ckpt/concept_emb/my_coco_48_base_17_cls_emb.pth \
    MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
    MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \
    MODEL.ENSEMBLE.TEST_CATEGORY_INFO "./datasets/coco_ovd_continue_cat_ids.json" \
    MODEL.ENSEMBLE.ALPHA 0.3 MODEL.ENSEMBLE.BETA 0.7 \
    OUTPUT_DIR output/eval
```
The results as follow:
[04/23 13:37:09] d2.evaluation.coco_evaluation INFO: Evaluation results for bbox: 
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 26.058 | 46.097 | 26.460 | 12.245 | 29.723 | 34.989 |
[04/23 13:37:09] d2.evaluation.coco_evaluation INFO: AP50_split_target AP: 0.3352322717304836
[04/23 13:37:09] d2.evaluation.coco_evaluation INFO: AP50_split_base AP: 0.5055024639541386
[04/23 13:37:09] d2.evaluation.coco_evaluation INFO: AP50_split_all AP: 0.4609702598341058






But when use trained got weights the results as follow:
[04/29 03:40:01] d2.evaluation.coco_evaluation INFO: Evaluation results for bbox: 
|  AP   |  AP50  |  AP75  |  APs  |  APm  |  APl  |
|:-----:|:------:|:------:|:-----:|:-----:|:-----:|
| 0.001 | 0.005  | 0.000  | 0.001 | 0.002 | 0.001 |
[04/29 03:40:01] d2.evaluation.coco_evaluation INFO: AP50_split_target AP: 1.0296307861183674e-05
[04/29 03:40:01] d2.evaluation.coco_evaluation INFO: AP50_split_base AP: 6.508518913363607e-05
[04/29 03:40:01] d2.evaluation.coco_evaluation INFO: AP50_split_all AP: 5.075578941622544e-05
[04/29 03:40:01] d2.evaluation.coco_evaluation INFO: Per-category bbox AP: 
| category   | AP    | category   | AP    | category     | AP    |
|:-----------|:------|:-----------|:------|:-------------|:------|
| person     | 0.050 | bicycle    | 0.000 | car          | 0.000 |
| motorcycle | 0.000 | airplane   | 0.000 | bus          | 0.000 |
| train      | 0.000 | truck      | 0.000 | boat         | 0.000 |
| bench      | 0.000 | bird       | 0.000 | cat          | 0.000 |
| dog        | 0.002 | horse      | 0.000 | sheep        | 0.000 |
| cow        | 0.000 | elephant   | 0.000 | bear         | 0.000 |
| zebra      | 0.000 | giraffe    | 0.000 | backpack     | 0.000 |
| umbrella   | 0.000 | handbag    | 0.000 | tie          | 0.000 |
| suitcase   | 0.000 | frisbee    | 0.000 | skis         | 0.000 |
| snowboard  | 0.000 | kite       | 0.000 | skateboard   | 0.000 |
| surfboard  | 0.000 | bottle     | 0.000 | cup          | 0.000 |
| fork       | 0.000 | knife      | 0.000 | spoon        | 0.000 |
| bowl       | 0.000 | banana     | 0.000 | apple        | 0.000 |
| sandwich   | 0.000 | orange     | 0.000 | broccoli     | 0.000 |
| carrot     | 0.000 | pizza      | 0.000 | donut        | 0.000 |
| cake       | 0.000 | chair      | 0.018 | couch        | 0.000 |
| bed        | 0.000 | toilet     | 0.000 | tv           | 0.000 |
| laptop     | 0.000 | mouse      | 0.000 | remote       | 0.005 |
| keyboard   | 0.000 | microwave  | 0.000 | oven         | 0.000 |
| toaster    | 0.000 | sink       | 0.000 | refrigerator | 0.000 |
| book       | 0.000 | clock      | 0.000 | vase         | 0.000 |
| scissors   | 0.000 | toothbrush | 0.000 |              |       |
[04/29 03:40:01] d2.evaluation.coco_evaluation INFO: avg inst: 100.0000 (4836 images)
[04/29 03:40:01] d2.engine.defaults INFO: Evaluation results for coco_2017_ovd_all_test in csv format:
[04/29 03:40:01] d2.evaluation.testing INFO: copypaste: Task: bbox
[04/29 03:40:01] d2.evaluation.testing INFO: copypaste: AP,AP50,AP75,APs,APm,APl,AP50_split_target,AP50_split_base,AP50_split_all
[04/29 03:40:01] d2.evaluation.testing INFO: copypaste: 0.0012,0.0051,0.0001,0.0007,0.0023,0.0009,0.0000,0.0001,0.0001

The code maybe somewhere use wrong trick, but I don't have method to deal with it.
</details>


### Results on LVIS-OVD
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Configs</th>
<th valign="bottom">APr</th>
<th valign="bottom">APc</th>
<th valign="bottom">APf</th>
<th valign="bottom">AP</th>
<!-- TABLE BODY -->
<!-- ROW: with LSJ -->
 <tr><td align="left"><a href="./sas_det/configs/ovd_lvis_R50_C4_ensemble_PLs.yaml">RN50-C4 as backbone</a></td>
<td align="center">20.1</td>
<td align="center">27.1</td>
<td align="center">32.9</td>
<td align="center">28.1</td>
</tr>
<!-- ROW: with out LSJ -->
 <tr><td align="left"><a href="./sas_det/configs/ovd_lvis_R50_C4_ensemble_PLs.yaml">RN50x4-C4 as backbone</a></td>
<td align="center">29.0</td>
<td align="center">32.3</td>
<td align="center">36.8</td>
<td align="center">33.5</td>
</tr>
</tbody></table>

<details>
<summary>
Evaluation with RN50-C4 as the backbone,
</summary>
  
```bash
python3 ./test_net.py \
    --num-gpus 8 \
    --eval-only \
    --config-file ./sas_det/configs/ovd_lvis_R50_C4_ensemble_PLs.yaml \
    MODEL.WEIGHTS ./pretrained_ckpt/sas_det/sas_det_lvis_r50.pth \
    MODEL.CLIP.OFFLINE_RPN_CONFIG ./sas_det/configs/regionclip/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
    MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866_lsj.pth \
    MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_866_base_cls_emb.pth \
    MODEL.CLIP.CONCEPT_POOL_EMB ./pretrained_ckpt/concept_emb/my_lvis_866_base_337_cls_emb.pth \
    MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb.pth \
    MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \
    MODEL.ENSEMBLE.TEST_CATEGORY_INFO "./datasets/lvis_ovd_continue_cat_ids.json" \
    MODEL.ENSEMBLE.ALPHA 0.33 MODEL.ENSEMBLE.BETA 0.67 \
    OUTPUT_DIR output/eval
```
</details>

<details>
<summary>
Evaluation with RN50x4-C4 as the backbone,
</summary>
  
```bash
python3 ./test_net.py \
    --num-gpus 8 \
    --eval-only \
    --config-file ./sas_det/configs/ovd_lvis_R50_C4_ensemble_PLs.yaml \
    MODEL.WEIGHTS ./pretrained_ckpt/sas_det/sas_det_lvis_r50x4.pth \
    MODEL.CLIP.OFFLINE_RPN_CONFIG ./sas_det/configs/regionclip/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
    MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866_lsj.pth \
    MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_866_base_cls_emb_rn50x4.pth \
    MODEL.CLIP.CONCEPT_POOL_EMB ./pretrained_ckpt/concept_emb/my_lvis_866_base_337_cls_emb_rn50x4.pth \
    MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb_rn50x4.pth \
    MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \
    MODEL.CLIP.TEXT_EMB_DIM 640 \
    MODEL.RESNETS.DEPTH 200 \
    MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
    MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION 18 \
    MODEL.ENSEMBLE.TEST_CATEGORY_INFO "./datasets/lvis_ovd_continue_cat_ids.json" \
    MODEL.ENSEMBLE.ALPHA 0.33 MODEL.ENSEMBLE.BETA 0.67 \
    OUTPUT_DIR output/eval
```
</details>



## Acknowledgement

This repository was built on top of [Detectron2](https://github.com/facebookresearch/detectron2), [RegionCLIP](https://github.com/microsoft/RegionCLIP), and [VLDet](https://github.com/clin1223/VLDet). We thank the effort from our community.
