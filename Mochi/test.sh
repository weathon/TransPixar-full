python3 cli.py \
    --prompt "A devil scorpionfish moves slowly across the seafloor, crawling with small, deliberate motions. The fish's body is excellently camouflaged, blending into the rocky, algae-covered environment with its mottled texture, muted colors, and irregular outline, making it very hard to see against the background." \
    --negative_prompt "highlighted, standing out, highly visible, vibrant tones, overexposed" \
    --num_inference_steps 64 \
    --num_frames 37 \
    --guidance_scale 6 \
    --lora_path "./mochi-rgba-lora-f37/checkpoint-1900.pt"

# after training negative prompt not working? mix in visible images?
    # --lora_path ~/wash/TransPixar-full/Mochi/mochi-rgba-lora-f37/checkpoint-2000.pt \
    # --height 480 \
    # --width 848 \
    # --negative_prompt "clearly visible, standing out, easy to spot, distinct edges, sharp outlines, high contrast, bright vibrant colors, unnatural hues, pixelated, low quality, text, overexposed, blurred body shape, artificial appearance, harsh borders" \
    