import base64
import os
from io import BytesIO

from openai import OpenAI
from PIL import Image


class GroundingModel:
    def __init__(self, url: str = "", api_key: str = ""):
        if not url:
            try:
                url = os.environ["GROUNDING_API_URL"]
            except KeyError:
                raise RuntimeError("GROUNDING_API_URL environment variable is not set")
        if not api_key:
            try:
                api_key = os.popen("eai login --profile default token").read().strip()
                if not api_key:
                    api_key = os.getenv("GROUNDING_API_KEY")
            except Exception:
                api_key = os.getenv("GROUNDING_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "Failed to get API key from 'eai login token' or GROUNDING_API_KEY environment variable"
                )
        self.vl_client = OpenAI(
            base_url=f"{url}/v1",
            api_key=api_key,
        )
        self.model = "osunlp/UGround-V1-7B"

    def get_coords(self, image: Image, element_description: str):
        img_width, img_height = image.size
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode()
        messages = self.format_openai_template(f"click {element_description}", base64_image)
        completion = self.vl_client.chat.completions.create(model=self.model, messages=messages, temperature=0.0)
        out_text = completion.choices[0].message.content
        str_x, str_y = out_text.strip("()").split(",")
        x = int(str_x.strip())
        y = int(str_y.strip())
        real_x = x / 1000 * img_width
        real_y = y / 1000 * img_height
        return real_x, real_y

    def format_openai_template(self, description: str, base64_image):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {
                        "type": "text",
                        "text": f"""
    Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

    - Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
    - If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
    - Your answer should be a single string (x, y) corresponding to the point of the interest.

    Description: {description}

    Answer:""",
                    },
                ],
            },
        ]
