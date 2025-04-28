python3 cli.py \
    --prompt "A snow leopard moves gracefully through a rocky, icy environment, its camouflaged fur blending seamlessly with the textures, colors, and shapes of the surrounding landscape. The animal's motion is fluid and silent, but due to its excellent camouflage, it remains partially concealed within the environment, making it difficult to spot at first glance.
" \
    --negative_prompt "the background is blue" \
    --num_inference_steps 10 \
    --num_frames 37 \
    --guidance_scale 9

    # --lora_path ~/wash/TransPixar-full/Mochi/mochi-rgba-lora-f37/checkpoint-2000.pt \
    # --height 480 \
    # --width 848 \