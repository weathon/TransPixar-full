python3 cli.py \
    --prompt 'This scene depicts a flatfish blending seamlessly into its environment, with its textured, mottled surface matching the surrounding textures, colors, and shapes, making it somewhat difficult to distinguish. The flatfish moves slowly across the seabed, subtly shifting position as it glides along, maintaining its camouflage. The motion is smooth and gentle, helping it remain hidden within its environment.' \
    --lora_path ~/wash/TransPixar-full/Mochi/mochi-rgba-lora-f37/checkpoint-2100.pt \
    --num_inference_steps 64 \
    --num_frames 37 \
    --guidance_scale 6 \