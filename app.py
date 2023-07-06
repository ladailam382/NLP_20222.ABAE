import gradio as gr
from inference import predict_abae, predict_acd, predict_cat, predict_uce

title = "Aspect Category Detection"
description = "Get aspect from given sentences."
article="Coming soon"

examples = [
  ["abae", "The bread is top notch as well."],
  ["acd", "The design and atmosphere is just as good."],
  ["cat", "The seafood is so fresh, but the hotdog is much more better"],
  ["uce", "The service is terrible"]
]

def fn(model_choice, input):
  if model_choice=="abae":
    return predict_abae(input)
  if model_choice=="acd":
    return predict_acd(input)
  if model_choice=="cat":
    return predict_cat(input)
  if model_choice=="uce":
    return predict_uce(input)

gr.Interface(fn, 
             [gr.Dropdown(["abae", "acd", "cat", "uce"], label="Model"), gr.Textbox(label="Input")],
             title=title, 
             description=description, 
             article=article,
             examples=examples,
             outputs=[gr.Textbox(label="Label"), gr.Label(label="Table")]).launch(server_port=8080)