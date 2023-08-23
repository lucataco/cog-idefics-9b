# Prediction interface for Cog ⚙️
from cog import BasePredictor, Input, Path
import torch
from PIL import Image
from transformers import IdeficsForVisionText2Text, AutoProcessor, BitsAndBytesConfig

MODEL_NAME = "HuggingFaceM4/idefics-9b"
MODEL_CACHE = "model-cache"

device = "cuda" if torch.cuda.is_available() else "cpu"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=["lm_head", "embed_tokens"],
        )
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            cache_dir="model-cache"
        )
        self.model = IdeficsForVisionText2Text.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=MODEL_CACHE
        )

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(
            description="Question about the image",
            default="What is in this image?",
        ),
        max_new_tokens: int = Input(
            description="max new tokens",
            default=8,
        ),
    ) -> str:
        """Run a single prediction on the model"""
        pil_image = Image.open(image)
        # We feed to the model an arbitrary sequence of text strings and images. Images can be either URLs or PIL Images.
        prompts = [
            [
                pil_image,
                "Question: "+prompt+" Answer:",
            ],
        ]

        bad_words = ["<image>", "<fake_token_around_image>"]
        bad_words_ids = self.processor.tokenizer(bad_words, add_special_tokens=False).input_ids
        eos_token = "</s>"
        eos_token_id = self.processor.tokenizer.convert_tokens_to_ids(eos_token)
        inputs = self.processor(prompts, return_tensors="pt").to("cuda")
        generated_ids = self.model.generate(
            **inputs, 
            eos_token_id=[eos_token_id], 
            bad_words_ids=bad_words_ids, 
            max_new_tokens=max_new_tokens,
            early_stopping=True
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
