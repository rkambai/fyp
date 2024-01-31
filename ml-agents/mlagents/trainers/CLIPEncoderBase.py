from onnxruntime import InferenceSession
from transformers import CLIPImageProcessor, CLIPTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

## model_path: fyp/Assets/Models/openai-clip-vit-large-patch14/model
## processor_path: fyp/Assets/Models/openai-clip-vit-large-patch14

class CLIPEncoderBase:
    def __init__(self, model_path: str, processor_path: str):
        self.model_path = model_path
        self.processor_path = processor_path
        self.session = None
        self.image_processor = None
        self.tokenizer = None

    def start_session(self):
        # Load CLIP model and processor
        self.session = InferenceSession(f"{self.model_path}.onnx")
        self.image_processor = CLIPImageProcessor.from_pretrained(f"{self.processor_path}")
        self.tokenizer = CLIPTokenizerFast.from_pretrained(f"{self.processor_path}")


    def run_inference(self, text, image):
        do_rescale = bool(np.max(image.flatten()) > 1)
        image = image.astype(np.int64)
        tokenized_inputs = self.tokenizer(text=[text])
        processed_image = self.image_processor.preprocess(images=image, return_tensors="np", padding=True, input_data_format = "channels_first", do_rescale=do_rescale)
        tokenized_inputs["pixel_values"] = processed_image.pixel_values
        logits_per_image, logits_per_text, text_embeds, image_embeds = self.session.run(output_names=["logits_per_image", "logits_per_text", "text_embeds", "image_embeds"], input_feed=dict(tokenized_inputs))
        self.logits_per_image = logits_per_image
        self.logits_per_text = logits_per_text
        self.text_embeds = text_embeds
        self.image_embeds = image_embeds
    
    def get_pairwise_similarity(self):
        return cosine_similarity(self.text_embeds, self.image_embeds)
