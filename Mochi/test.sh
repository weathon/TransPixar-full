python3 cli.py \
    --prompt "A slender pipefish gently glides among swaying sea grass, its elongated body perfectly matching the texture, color, and shape of the underwater plants. The pipefish\'s subtle movement mimics the undulating currents, making it blend seamlessly into its aquatic surroundings as its delicate pattern and muted hues melt into the background." \
    --num_inference_steps 64 \
    --num_frames 37 \
    --guidance_scale 6 \
    --lora_path "./mochi-rgba-lora-f37/checkpoint-3000.pt"

# after training negative prompt not working? mix in visible images?
    # --lora_path ~/wash/TransPixar-full/Mochi/mochi-rgba-lora-f37/checkpoint-2000.pt \
    # --height 480 \
    # --width 848 \
    # --negative_prompt "clearly visible, standing out, easy to spot, distinct edges, sharp outlines, high contrast, bright vibrant colors, unnatural hues, pixelated, low quality, text, overexposed, blurred body shape, artificial appearance, harsh borders" \
    