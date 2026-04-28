import torch
import scipy.io.wavfile
import os
import time
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import json

def evaluate_prompts():
    model_id = "facebook/musicgen-small"
    print(f"Loading {model_id}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id)
    model.to(device)
    
    # Baseline vs Improved prompts
    scenarios = [
        {
            "id": 1,
            "baseline": "electronic music",
            "improved": "Fast-paced synthwave electronic music with a driving bassline, retro 80s drums, and a catchy lead synth melody",
        },
        {
            "id": 2,
            "baseline": "rock music",
            "improved": "Upbeat punk rock music with heavily distorted guitars, aggressive drum beats, and a rebellious high-energy vibe",
        },
        {
            "id": 3,
            "baseline": "relaxing music",
            "improved": "Ambient spa music featuring a soft grand piano, gentle ocean waves in the background, and slow ethereal strings",
        }
    ]
    
    os.makedirs("outputs", exist_ok=True)
    results = []
    
    duration = 5  # 5 seconds for evaluation to save time
    max_new_tokens = int(duration * 50)
    
    for scenario in scenarios:
        for prompt_type in ["baseline", "improved"]:
            prompt = scenario[prompt_type]
            print(f"\nProcessing Scenario {scenario['id']} - {prompt_type.capitalize()}")
            print(f"Prompt: {prompt}")
            
            inputs = processor(text=[prompt], padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            start_time = time.time()
            with torch.no_grad():
                audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generation_time = time.time() - start_time
            
            sampling_rate = model.config.audio_encoder.sampling_rate
            audio_array = audio_values[0, 0].cpu().numpy()
            
            filename = f"outputs/scenario_{scenario['id']}_{prompt_type}.wav"
            scipy.io.wavfile.write(filename, rate=sampling_rate, data=audio_array)
            
            print(f"Saved to {filename}. Latency: {generation_time:.2f}s")
            
            results.append({
                "scenario_id": scenario["id"],
                "type": prompt_type,
                "prompt": prompt,
                "latency_seconds": round(generation_time, 2),
                "filename": filename
            })
            
    # Save results summary
    with open("outputs/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nEvaluation complete! Please listen to the generated files in the 'outputs' folder and compare the realism, creativity, and diversity.")

if __name__ == "__main__":
    evaluate_prompts()
