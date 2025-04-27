python3 cli.py \
    --prompt "A crab, expertly camouflaged to blend in with the rocky and sandy ocean floor due to its texture, color, and shape, moves slowly and carefully across the seabed. Its subtle movements are barely noticeable as it seamlessly merges with its surroundings, making it difficult to spot among the rocks and debris. The crab's body mimics the environment, rendering it almost invisible while it traverses the marine landscape." \
    --lora_path ~/wash/TransPixar-full/Mochi/mochi-rgba-lora-f37/checkpoint-2100.pt \
    --num_inference_steps 10 \
    --num_frames 48 \
    --guidance_scale 6 \