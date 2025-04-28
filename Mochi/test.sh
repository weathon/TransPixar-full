python3 cli.py \
    --prompt "A devil scorpionfish moves slowly across the seafloor, crawling with small, deliberate motions. The fish's body is excellently camouflaged, blending into the rocky, algae-covered environment with its mottled texture, muted colors, and irregular outline, making it very hard to see against the background." \
    --negative_prompt "clearly visible, highlighted, prominent, obvious, motionless" \
    --num_inference_steps 64 \
    --num_frames 37 \
    --guidance_scale 9 \
    --lora_path "./mochi-rgba-lora-f37/checkpoint-3600.pt"
    

    # --lora_path ~/wash/TransPixar-full/Mochi/mochi-rgba-lora-f37/checkpoint-2000.pt \
    # --height 480 \
    # --width 848 \