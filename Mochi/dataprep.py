
# %%
import os

from PIL import Image
import os
import imageio.v2 as imageio
import numpy as np
import cv2

path = "/home/wg25r/fastdata/fullmoca/MoCA-Video-Train/"
rgb_dir = "./video_rgb"
alpha_dir = "./video_alpha"

if os.path.exists(rgb_dir):
    os.system(f"rm -rf {rgb_dir}")
os.makedirs(rgb_dir)

if os.path.exists(alpha_dir):
    os.system(f"rm -rf {alpha_dir}")
os.makedirs(alpha_dir)

# for video in os.listdir(path):
def process_video(video):
    frame_dir = os.path.join(path, video, "Frame")
    gt_dir = os.path.join(path, video, "GT")
    if not (os.path.isdir(frame_dir) and os.path.isdir(gt_dir)):
        return
    frames = sorted(f for f in os.listdir(frame_dir) if f.lower().endswith(".jpg"))
    if len(frames) < 37:
        print(f"Video {video} has less than 37 frames, skipping...")
        return

    for idx in range(len(frames) // 37):
        seg_frames = frames[idx * 37:(idx + 1) * 37]

        rgb_out_path = os.path.join(rgb_dir, f"{video}_{idx:03d}.mp4")
        writer_rgb = imageio.get_writer(rgb_out_path, fps=24)
        for i, f in enumerate(seg_frames):
            img = imageio.imread(os.path.join(frame_dir, f))
            img_resized = Image.fromarray(img).resize((416, 240))
            writer_rgb.append_data(np.array(img_resized).astype(np.uint8))
        writer_rgb.close()

        alpha_out_path = os.path.join(alpha_dir, f"{video}_{idx:03d}.mp4")
        writer_alpha = imageio.get_writer(alpha_out_path, fps=24)
        for i, f in enumerate(seg_frames):
            alpha_name = os.path.splitext(f)[0] + '.png'
            mask = np.array(imageio.imread(os.path.join(gt_dir, alpha_name)))
            mask_resized = cv2.resize(mask, (416, 240), interpolation=cv2.INTER_NEAREST)
            img = np.array(imageio.imread(os.path.join(frame_dir, f)))
            img = cv2.resize(img, (416, 240), interpolation=cv2.INTER_NEAREST)
            background_color = (img[mask_resized < 128]).mean(axis=0)
            foreground_color = (img[mask_resized > 128]).mean(axis=0)
            foreground = np.where(mask_resized[:,:,None] > 0, foreground_color, background_color).astype(int)
            foreground = np.clip(foreground, 0, 255).astype(np.uint8)
                                                
            writer_alpha.append_data(foreground.astype(np.uint8))
        writer_alpha.close()
    return foreground.astype(int), np.array(img_resized).astype(int)


# %%
videos = os.listdir(path)

a, b = process_video(videos[0]) 

import matplotlib.pyplot as plt
plt.imshow(a)


from multiprocessing import Pool
pool = Pool(processes=64)
pool.map(process_video, videos)
pool.close()
pool.join() 

print("Finished positive samples")




# %%
import os
# animal_videos = ["bear", "blackswan", "camel", "cows", "dog", "dog-agility", "elephant", "flamingo", "goat", "horsejump-high", "horsejump-low", "mallard-fly", "mallard-water", "rhino"]
animal_videos = os.listdir("/mnt/fastdata/DAVIS16/JPEGImages")
path = "/mnt/fastdata/DAVIS16"
# for animal in animal_videos: huxikunduzihuxikunkk
#     print(os.listdir(os.path.join(path, animal)))

# %%
videos = os.listdir("/home/wg25r/fastdata/fullmoca/MoCA-Video-Train/")

# %%
set(animal_videos).intersection(set(videos))

# %%
print(animal_videos)

# %%
# animal_videos

# %%
from PIL import Image
import os
import imageio.v2 as imageio
import numpy as np
import random
import cv2

rgb_dir = "./video_rgb"
alpha_dir = "./video_alpha"


# for video in os.listdir(path):
def process_video_normal(video):
    frame_dir = os.path.join(path, "JPEGImages", video)
    gt_dir = os.path.join(path, "Annotations", "480p", video)
    if not (os.path.isdir(frame_dir) and os.path.isdir(gt_dir)):
        return
    frames = sorted(f for f in os.listdir(frame_dir) if f.lower().endswith(".jpg"))
    if len(frames) < 37:
        print(f"Video {video} has less than 37 frames, skipping...")
        return

    for idx in range(len(frames) // 37):
        seg_frames = frames[idx * 37:(idx + 1) * 37]

        rgb_out_path = os.path.join(rgb_dir, f"neg_{video}_{idx:03d}.mp4")
        writer_rgb = imageio.get_writer(rgb_out_path, fps=24)
        for i, f in enumerate(seg_frames):
            img = imageio.imread(os.path.join(frame_dir, f))
            img_resized = Image.fromarray(img).resize((416, 240))
            writer_rgb.append_data(np.array(img_resized).astype(np.uint8))
        writer_rgb.close()

        alpha_out_path = os.path.join(alpha_dir, f"neg_{video}_{idx:03d}.mp4")
        writer_alpha = imageio.get_writer(alpha_out_path, fps=24)
        for i, f in enumerate(seg_frames):
            alpha_name = os.path.splitext(f)[0] + '.png'
            mask = np.array(imageio.imread(os.path.join(gt_dir, alpha_name)))
            mask_resized = cv2.resize(mask, (416, 240), interpolation=cv2.INTER_NEAREST)
            img = np.array(imageio.imread(os.path.join(frame_dir, f)))
            img = cv2.resize(img, (416, 240), interpolation=cv2.INTER_NEAREST)
            background_color = img[mask_resized < 128].mean(axis=0)
            foreground_color = (img[mask_resized > 128]).mean(axis=0)
            foreground = np.where(mask_resized[:,:,None] > 0, foreground_color, background_color).astype(int)
                        
            writer_alpha.append_data(foreground.astype(np.uint8))
        writer_alpha.close()
        
    return foreground.astype(int), np.array(img_resized).astype(int)


def process_video_highlight(video):
    frame_dir = os.path.join(path, "JPEGImages", video)
    gt_dir = os.path.join(path, "Annotations", "480p", video)
    if not (os.path.isdir(frame_dir) and os.path.isdir(gt_dir)):
        return
    frames = sorted(f for f in os.listdir(frame_dir) if f.lower().endswith(".jpg"))
    if len(frames) < 37:
        print(f"Video {video} has less than 37 frames, skipping...")
        return

    for idx in range(len(frames) // 37):
        seg_frames = frames[idx * 37:(idx + 1) * 37]
        for version in range(3):
            factor = random.random() * 3
            rgb_out_path = os.path.join(rgb_dir, f"neg_hl_{video}_{idx:03d}_{version}.mp4")
            writer_rgb = imageio.get_writer(rgb_out_path, fps=24)
            for i, f in enumerate(seg_frames):
                img = imageio.imread(os.path.join(frame_dir, f))
                img_resized = Image.fromarray(img).resize((416, 240)) 
                img_resized = np.array(img_resized).astype(int)
                mask = np.array(imageio.imread(os.path.join(gt_dir, os.path.splitext(f)[0] + '.png')))
                mask_resized = cv2.resize(mask, (416, 240), interpolation=cv2.INTER_NEAREST)
                img_resized = np.where(mask_resized[:,:,None] > 0, img_resized * factor, img_resized).astype(int)            
                img_resized = np.clip(img_resized, 0, 255).astype(np.uint8)
                writer_rgb.append_data(np.array(img_resized).astype(np.uint8))
                
            writer_rgb.close()

            alpha_out_path = os.path.join(alpha_dir, f"neg_hl_{video}_{idx:03d}_{version}.mp4")
            writer_alpha = imageio.get_writer(alpha_out_path, fps=24)
            for i, f in enumerate(seg_frames):
                alpha_name = os.path.splitext(f)[0] + '.png'
                mask = np.array(imageio.imread(os.path.join(gt_dir, alpha_name)))
                mask_resized = cv2.resize(mask, (416, 240), interpolation=cv2.INTER_NEAREST)
                img = np.array(imageio.imread(os.path.join(frame_dir, f)))
                img = cv2.resize(img, (416, 240), interpolation=cv2.INTER_NEAREST)
                background_color = (img[mask_resized < 128]).mean(axis=0)
                foreground_color = (img[mask_resized > 128]).mean(axis=0)
                foreground = np.where(mask_resized[:,:,None] > 0, foreground_color, background_color).astype(int)
                foreground = np.clip(foreground, 0, 255).astype(np.uint8)
                
                # img_clipped = np.where(mask_resized[:,:,None] > 0, 255, 0)
                            
                writer_alpha.append_data(foreground.astype(np.uint8))
            writer_alpha.close()
            
    return foreground.astype(int), np.array(img_resized).astype(int)


def process_video(video):
    process_video_highlight(video)
    process_video_normal(video)

process_video(animal_videos[0])

# %%
from multiprocessing import Pool
pool = Pool(processes=64)
pool.map(process_video, animal_videos)
pool.close()
pool.join()




print("Finished negative samples")



from openai import OpenAI
from PIL import Image
import io
import base64
from pydantic import BaseModel
from typing import List, Optional

class Prompt(BaseModel):
    pos_prompt: str
  

positive_gpt_prompt = "Describe in 2-3 setence this image such that the prompt will be used to generate an video. Mention motion, the object might be hard to see. the filename is %s, filename might has other things than the animal. Do not mention filename just the animal name. The object might be hard to see (mention this) and blended into the envirement. Mention the animal is camflague and blend into the envirement for texture, color, and shape. Mention the animal's motion, but do not mention this makes it easy to spot. At the same time, provide a negative prompt, tell what the model should not generate, could be: clearly visible, standing out, easy to spot, obvious, distinct, high contrast, sharp outline, border, vibrant colors, unnatural colors, unnature bodies, blurry, low quality, pixelated, text, over exposure, etc. Do just just copy paste this, write it based on specific video. Both prompt should be description and not action requiring. Do not mention positive part in negative prompt and vice versa. Negative prompt should be description of what not to be in a positive way, e.g. 'ugly face', instead of 'do not generate ugly face'. Negative prompt should be information dense, not detialed, and just a list of words. Use simple language."

negative_gpt_prompt = "Describe in 2-3 setence this image such that the prompt will be used to generate an video. Mention motion. the filename is %s, filename might has other things than the main object. Do not mention filename just the main object name. The object is clearly stand out and maybe highlighted or contrasted (if the filename has 'hl_' at the begining, if so mention this) and standout into the envirement (mention this always). Mention the object's motion. Use simple language and vocab besides the name of the animal" 


    
def capture_image(imgs, filename):
    client = OpenAI()
    base64s = []
    for img in imgs:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_data = buffer.getvalue()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{base64_image}"
        base64s.append(image_url)
        
    response = client.beta.chat.completions.parse(
    model="gpt-4.1",
    messages=[
        {
        "role": "user",
        "content": [
            *[{
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
            } for image_url in base64s],
            {
            "type": "text",
            "text": positive_gpt_prompt % filename if "neg_" not in filename else negative_gpt_prompt % filename
            }
        ]
        }, 
    ],
    response_format=Prompt,
    )
    return response.choices[0].message.content

import os
import cv2
videos = os.listdir("./video_rgb")

def get_prompt(video):
    video_path = os.path.join("./video_alpha", video)
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    # get first, middle, and last frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2
    for i in [0, middle_frame, total_frames - 1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    # Convert frames to PIL images
    imgs = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    # Capture image and get prompt
    prompt = capture_image(imgs, video) 
    
    with open(os.path.join("./video_rgb", video.replace(".mp4", ".txt")), "w") as f:
        f.write(prompt)
    print(prompt)
    return prompt
videos = os.listdir("./video_rgb")
print(len(videos))
# use a pool of 30 processes
from multiprocessing import Pool
if __name__ == "__main__":
    with Pool(30) as p:
        prompts = p.map(get_prompt, videos)
    print(prompts)

# get_prompt("crab_1_000.mp4")