# AI Music Generation System - Quiz Challenge 2

This repository contains the implementation of an AI Music and Sound Generation system using Hugging Face's `facebook/musicgen-small` foundation model.

## Problem Description & Business Value
Creating high-quality music is time-consuming and requires specialized skills. This system demonstrates how generative AI can be used to rapidly prototype and generate custom background music and sound effects for games, podcasts, and video production, drastically reducing costs and time-to-market.

## Pipeline Architecture
1. **Frontend:** Gradio web interface (`app.py`).
2. **Model:** `facebook/musicgen-small` loaded via Hugging Face `transformers`.
3. **Processing:** Text prompts are tokenized and passed to the auto-regressive transformer.
4. **Output:** The model generates audio tokens which are decoded into a `.wav` file and served back to the user.

## Requirements and Installation

Make sure you have Python installed. Then, install the required libraries:

```bash
pip install -r requirements.txt
```

## How to Run the Application (Demo UI)

To start the Gradio web interface:

```bash
python app.py
```

This will launch a local web server. Open the provided URL in your browser to test different text prompts and generate music.

## How to Run the Evaluation Script

To compare baseline vs. improved prompts and generate sample outputs:

```bash
python evaluate.py
```

The generated `.wav` files and a JSON report containing latency metrics will be saved in the `outputs/` folder.

## AI Tools Disclosure
- **Google Gemini:** Assisted in writing the Python application code (`app.py`), the evaluation script (`evaluate.py`), the `presentation_outline.md`, and this `README.md`.