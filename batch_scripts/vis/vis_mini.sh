# Set the path to save video
OUTPUT_DIR='/svl/u/mjlbach/VideoMAE_test'
# path to video for visualization
# path to pretrain model
MODEL_PATH='/svl/u/mjlbach/ego_vae_mini/checkpoint-229.pth'
OUTPUT_DIR='/svl/u/mjlbach/ego_vae_mini_macondo_test'
MODEL_PATH='/svl/u/mjlbach/ego_vae_macondo/checkpoint-3.pth'
#VIDEO_PATH='/vision/group/ego4d/v1/full_scale/40a4de2a-a6e6-46eb-a7cf-cd79fae7c5f8.mp4'
#VIDEO_PATH='/vision/group/ego4d/v1/full_scale/c8b61175-cfa4-44f9-a88b-76609ffa4ca3.mp4'
#VIDEO_PATH='/vision/group/ego4d/v1/full_scale/89e7168c-f17f-4acd-aceb-740ebd0b760e.mp4'
#VIDEO_PATH='/vision/group/ego4d/v1/full_scale/23aeabe2-1e22-40c8-9c6d-89424e5ec4d1.mp4'
#VIDEO_PATH='/vision/group/ego4d/v1/full_scale/a0b158a7-745c-40ce-a5b2-d341feefdac0.mp4'
#VIDEO_PATH='/vision/group/ego4d/v1/full_scale/6c0d1d45-19fb-4992-8d0f-1a4f226b19d3.mp4'
#VIDEO_PATH='/vision/group/ego4d/v1/full_scale/a3103e6e-311f-4d4f-8eb6-7278da3dd3e2.mp4'
#VIDEO_PATH='/vision/group/ego4d/v1/full_scale/f5ecb86e-ab05-41e2-8775-cf18c0202616.mp4'
#VIDEO_PATH='/vision/group/ego4d/v1/full_scale/3a839ee1-2391-43bb-a253-c2c45d753bac.mp4'
#VIDEO_PATH='/vision/group/ego4d/v1/full_scale/f262e5b6-6bf0-42b3-b8a6-2ea1b61c75af.mp4'
#VIDEO_PATH='/vision/group/ego4d/v1/full_scale/d752fc52-9592-4cb5-910b-d09bb39261fd.mp4'
VIDEO_PATH='/vision/group/ego4d/v1/full_scale/df9c2151-58c3-471c-8e91-caa8afc61c65.mp4'
#VIDEO_PATH='/vision/group/ego4d/v1/full_scale/bea8cf53-5c59-4169-a161-b58aedb9f12a.mp4'

VIDEO_PATH='/svl/u/mjlbach/output.mp4'

#VIDEO_PATH='/vision/group/ego4d/v1/full_scale/001e3e4e-2743-47fc-8564-d5efd11f9e90.mp4'
#VIDEO_PATH='/vision/group/ego4d/v1/full_scale/004a1802-c546-4dcc-86ba-bf1080077017.mp4'

python3 run_videomae_vis.py \
    --mask_ratio 0.9 \
    --mask_type tube \
    --decoder_depth 4 \
    --sampling_rate 4 \
    --model pretrain_videomae_base_patch16_224 \
    ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}
