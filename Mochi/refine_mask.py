from sam2.build_sam import build_sam2_video_predictor
sam2_checkpoint = "/home/wg25r/grounded_mog/.sam2/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cuda")