import gradio as gr
import torch
import scipy.io.wavfile
import os
import time
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# Initialize model
model_id = "facebook/musicgen-small"
print(f"Loading {model_id}...")
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id)
    model.to(device)
except Exception as e:
    print(f"Error loading model: {e}")

def generate_music(prompt, duration=5):
    # MusicGen outputs 50 steps per second of audio.
    max_new_tokens = int(duration * 50)
    
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    )
    
    # move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    start_time = time.time()
    
    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    generation_time = time.time() - start_time
    
    # Extract audio array and sampling rate
    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_array = audio_values[0, 0].cpu().numpy()
    
    # Save temporarily to a file
    output_path = f"outputs/temp_{int(time.time())}.wav"
    os.makedirs("outputs", exist_ok=True)
    scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio_array)
    
    return output_path, f"Generation Time: {generation_time:.2f} seconds"

# Define Gradio interface
with gr.Blocks(title="AI Music Generation", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎵 AI Music & Sound Generation System")
    gr.Markdown("This application uses **facebook/musicgen-small** to generate music and sound effects from text prompts. It is part of the Quiz Challenge 2.")
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(label="Enter a text prompt", placeholder="e.g., '80s driving pop song with heavy guitars and brass'", lines=3)
            duration_input = gr.Slider(minimum=2, maximum=15, value=5, step=1, label="Duration (seconds)")
            generate_button = gr.Button("Generate Music", variant="primary")
        
        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Generated Audio", type="filepath")
            stats_output = gr.Textbox(label="Generation Stats", interactive=False)
            
    gr.Examples(
        examples=[
            ["A chill lofi hip hop beat with a smooth saxophone loop", 5],
            ["Heavy metal guitar riff with double bass drums", 5],
            ["Orchestral battle music with intense strings and horns", 8],
            ["A relaxing piano melody with nature sounds in the background", 10]
        ],
        inputs=[prompt_input, duration_input]
    )
            
    generate_button.click(
        fn=generate_music,
        inputs=[prompt_input, duration_input],
        outputs=[audio_output, stats_output]
    )

if __name__ == "__main__":
    print("Starting Gradio server...")
    demo.launch()
