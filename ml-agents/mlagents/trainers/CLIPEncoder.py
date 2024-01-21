from onnxruntime import InferenceSession
from transformers import CLIPProcessor
from sklearn.metrics.pairwise import cosine_similarity

## model_path: fyp/Assets/Models/openai-clip-vit-large-patch14/model
## processor_path: fyp/Assets/Models/openai-clip-vit-large-patch14

class CLIPEncoder:
    def __init__(self, model_path: str, processor_path: str):
        self.model_path = model_path
        self.processor_path = processor_path
        self.session = None
        self.processor = None

    def start_session(self):
        # Load CLIP model and processor
        self.session = InferenceSession(f"{self.model_path}.onnx")
        self.processor = CLIPProcessor.from_pretrained(f"{self.processor_path}")

    def run_inference(self, text, image):
        inputs = self.processor(text=[text], images=image, return_tensors="np", padding=True)
        logits_per_image, logits_per_text, text_embeds, image_embeds = self.session.run(output_names=["logits_per_image", "logits_per_text", "text_embeds", "image_embeds"], input_feed=dict(inputs))
        self.logits_per_image = logits_per_image
        self.logits_per_text = logits_per_text
        self.text_embeds = text_embeds
        self.image_embeds = image_embeds
    
    def get_pairwise_similarity(self):
        return cosine_similarity(self.text_embeds, self.image_embeds)
