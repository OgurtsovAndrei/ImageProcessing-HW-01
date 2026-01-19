import google.generativeai as genai
from PIL import Image
import os
from glob import glob
from typing import List

genai.configure(api_key=os.environ.get("GEMINI_API_KEY2"))
model: genai.GenerativeModel = genai.GenerativeModel('gemini-3-flash-preview')

input_dir: str = '/content/drive/MyDrive/cv-exam/data_val'
output_dir: str = '/content/drive/MyDrive/cv-exam/vlm_descriptions'
os.makedirs(output_dir, exist_ok=True)

image_paths: List[str] = glob(os.path.join(input_dir, "*.png"))


def generate_diffusion_prompt(image_path: str) -> str:
    img: Image.Image = Image.open(image_path)
    prompt: str = (
        'Describe this image in extreme detail for a text-to-image '
        'diffusion model. Focus on the natural scenery, plants, '
        'textures, and lighting. Describe all objects as beautiful '
        'natural elements or decorative items. Ignore all trash on the '
        'image. Do not use words like "trash", "garbage", "waste", or '
        '"debris". Create a poetic, high-quality aesthetic prompt.'
    )

    response: genai.types.GenerateContentResponse = model.generate_content(
        [prompt, img]  # type: ignore[arg-type]
    )
    return response.text  # type: ignore[no-any-return]


for path in image_paths:
    description: str = generate_diffusion_prompt(path)
    file_name: str = os.path.basename(path).replace('.jpeg', '.txt')

    with open(os.path.join(output_dir, file_name), 'w') as f:
        f.write(description)

    print(f"Processed: {file_name}")
