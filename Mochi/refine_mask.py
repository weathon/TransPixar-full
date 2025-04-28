from sam2.build_sam import build_sam2_video_predictor
import os
from PIL import Image
import numpy as np


import imageio
import glob

sam2_checkpoint = "/home/wg25r/grounded_mog/.sam2/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cuda")
videos = os.listdir("/home/wg25r/fastdata/fullmoca/MoCA-Video-Train/")


def mask_to_box(mask):
    mask = np.array(mask) > 128
    y_indices, x_indices = np.where(mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return 0, 0, 0, 0
    x_min = np.min(x_indices)
    x_max = np.max(x_indices)
    y_min = np.min(y_indices)
    y_max = np.max(y_indices)
    return x_min, y_min, x_max, y_max
    

for video in videos[:1]:
    inference_state = predictor.init_state(video_path=os.path.join("/home/wg25r/fastdata/fullmoca/MoCA-Video-Train/", video, "Frame"))
    predictor.reset_state(inference_state)
    for i, frames in enumerate(os.listdir(os.path.join("/home/wg25r/fastdata/fullmoca/MoCA-Video-Train/", video, "Frame"))):
        frame_path = os.path.join("/home/wg25r/fastdata/fullmoca/MoCA-Video-Train/", video, "Frame", frames)
        mask_path = os.path.join("/home/wg25r/fastdata/fullmoca/MoCA-Video-Train/", video, "GT", frames.replace("jpg", "png"))
        frame_id = int(frames.split(".")[0])
        if frame_id % 5 == 0:
            original_mask = Image.open(mask_path)      
            original_box = mask_to_box(original_mask)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=i,
                obj_id=1,
                box=original_box,
            )
            
            # _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            #     inference_state=inference_state,
            #     frame_idx=i,
            #     obj_id=1,
            #     mask=np.array(original_mask),
            # )
              
    video_segments = {} 
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    # print("video_segments", video_segments)
    writer = imageio.get_writer('test.mp4', fps=5)
    for i in video_segments.keys():
        mask = video_segments[i][1]
        mask = np.array(mask) > 0.5
        mask = mask.astype(np.uint8)[0] * 255
        frame = Image.open(os.path.join("/home/wg25r/fastdata/fullmoca/MoCA-Video-Train/", video, "Frame", f"{i:05d}.jpg"))
        frame = np.array(frame)
        mask = np.stack([mask] * 3, axis=-1)
        frame = frame * 0.5 + mask * 0.5
        frame = frame.astype(np.uint8)
        writer.append_data(frame)
    writer.close()
            