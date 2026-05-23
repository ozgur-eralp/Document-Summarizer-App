# *** COPYRIGHT NOTICE ***
# © 2025 Ozgur Eralp. All Rights Reserved.

import gradio as gr
import asyncio

from pdf_summarizer import summarize_pdf, display_pdf_preview, get_summarizer

css = """
/* Target the pagination controls (adjust selectors if needed) */
.gr-dataset-pagination-container .page-number,
.gr-dataset-pagination-container .page-btn { /* Include page-btn as well */
    font-size: 16px !important; /* Adjust font size as needed */
}

/* Optional: Style the "of N" text */
.gr-dataset-pagination-container .of-pages {
    font-size: 14px !important; /* Adjust font size as needed */
}

#bottom-aligned-row {
    display: flex;
    align-items: flex-end; /* Aligns items at the bottom */
}

footer {visibility: hidden}
"""

with gr.Blocks(delete_cache=(86400, 86400)) as myapp:
    myapp.title = "Analysis App"

    gr.Markdown("# File Summarizer")

    loading_status = gr.Markdown("## ⚠️ System Initializing: Loading AI models. Please wait...")
    
    with gr.Row(visible=False):
        max_chunk_length_slider = gr.Slider(minimum=512, maximum=4096, step=128, value=2048, label="Max Chunk Length")
        max_summary_length_slider = gr.Slider(minimum=25, maximum=250, step=25, value=100, label="Max Summary Length")

    summary_length_option = gr.Radio(choices=["short", "normal", "long"], value="normal", label="Summary Length")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(file_types=[".pdf"], label="Upload PDF")
            pdf_preview_output = gr.HTML(label="PDF Preview")

    with gr.Row(elem_id="bottom-aligned-row"):
        submit_btn = gr.Button("Summarize", variant="primary", interactive=False)

    audio_output = gr.Audio(label="Summary Audio", type="filepath")
    output_text = gr.Textbox(label="Summarized Text")
    
    submit_btn.click(summarize_pdf, inputs=[file_input, max_chunk_length_slider, max_summary_length_slider], outputs=[output_text, audio_output])
    file_input.upload(display_pdf_preview, inputs=file_input, outputs=pdf_preview_output)

    def update_sliders(summary_option):
        if summary_option == "short":
            return 3072, 80
        elif summary_option == "normal":
            return 2048, 100
        else:
            return 1024, 150
    summary_length_option.change(update_sliders, inputs=summary_length_option, outputs=[max_chunk_length_slider, max_summary_length_slider])

    def init_app():
        try:
            # Trigger the lazy loader
            get_summarizer()
            return gr.update(visible=False), gr.update(interactive=True)
        except Exception as e:
            return gr.update(value=f"## ❌ Error loading model: {e}"), gr.update(interactive=False)

    myapp.load(init_app, outputs=[loading_status, submit_btn])

myapp.launch(favicon_path="favicon.ico", server_port=5000, css=css)
