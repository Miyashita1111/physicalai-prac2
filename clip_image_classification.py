from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch
# CLIPモデルをtransformersライブラリを用いてダウンロード
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# 画像の取得

url = "https://huggingface.co/datasets/huggingface/documentation-
images/resolve/main/cats.png"

image = Image.open(requests.get(url, stream=True).raw)
# 自然言語での候補を準備
texts = ["a photo of a cat", "a photo of a dog", "a photo of a robot"]
# 前処理と推論
inputs = processor(text=texts, images=image, return_tensors="pt",
padding=True)
with torch.no_grad():
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
# 結果の表示
for text, prob in zip(texts, probs[0]):
print(f"{text}: {prob.item():.4f}")
