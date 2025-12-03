import base64
import json
import os
from io import BytesIO
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

SUPPORTED_EXT = [".pdf", ".jpg", ".jpeg", ".png"]


# -------------------------------------------------------------------
# NEW: Convert PDF → images in memory (NO SAVE TO DISK)
# -------------------------------------------------------------------
def pdf_to_images_in_memory(pdf_path):
    from pdf2image import convert_from_path
    print("Converting PDF to in-memory images...")

    pil_images = convert_from_path(pdf_path, dpi=300)
    image_bytes_list = []

    for img in pil_images:
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        image_bytes_list.append(buffer.read())

    print(f"Loaded {len(image_bytes_list)} page(s) in memory")
    return image_bytes_list


# -------------------------------------------------------------------
# Batch-safe menu extraction for each image
# -------------------------------------------------------------------
def extract_menu_from_image_bytes(image_bytes, groq_api_key, max_batches=20):
    client = Groq(api_key=groq_api_key)

    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    common_rules = """
You extract menu items from an image.

Rules:
1. NEVER merge items.
2. One price = one item.
3. If a name has multiple prices, split them into separate items.
4. If a line contains multiple menu entries, split them.
5. Ignore noise, ads, contact info.
6. Output ONLY a JSON array of:
   { "name": "string", "price": number }
"""

    all_items = []

    for batch_num in range(1, max_batches + 1):
        prompt = f"""
{common_rules}

Extract batch #{batch_num}.
If no more items remain in the image, return [].
Return ONLY JSON array.
"""

        resp = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.2,
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}"
                            },
                        },
                    ],
                }
            ],
        )

        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            batch_items = json.loads(raw)
        except Exception:
            print(f"Batch {batch_num} returned invalid JSON. Skipping.")
            continue

        if not batch_items:
            break  # no more data

        all_items.extend(batch_items)

    return all_items


# -------------------------------------------------------------------
# Save JSON only
# -------------------------------------------------------------------
def save_menu_json(menu_data, input_path, output_filename=None):
    folder = os.path.dirname(input_path) or "."
    if output_filename is None:
        name = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{name}_extracted.json"

    output_path = os.path.join(folder, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(menu_data, f, indent=2, ensure_ascii=False)

    print(f"Saved JSON → {output_path}")
    return output_path


# -------------------------------------------------------------------
# Scan "menu" folder
# -------------------------------------------------------------------
def get_menu_files_from_folder(folder="menu"):
    if not os.path.exists(folder):
        raise ValueError(f"Menu folder not found: {folder}")

    files = []
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        ext = os.path.splitext(f)[1].lower()
        if ext in SUPPORTED_EXT:
            files.append(path)

    if not files:
        raise ValueError("No supported menu files found in /menu folder.")

    return files


# -------------------------------------------------------------------
# Main Script
# -------------------------------------------------------------------
if __name__ == "__main__":

    menu_files = get_menu_files_from_folder("menu")
    print(f"Found {len(menu_files)} menu file(s).\n")

    for menu_file in menu_files:

        print("=" * 60)
        print(f"Processing: {menu_file}")
        print("=" * 60)

        # Convert PDF to images in memory OR load image directly
        if menu_file.lower().endswith(".pdf"):
            images_bytes = pdf_to_images_in_memory(menu_file)
        else:
            with open(menu_file, "rb") as f:
                images_bytes = [f.read()]

        all_items = []

        for idx, img_bytes in enumerate(images_bytes, 1):
            print(f"\nExtracting page {idx}/{len(images_bytes)}...")
            page_items = extract_menu_from_image_bytes(img_bytes, GROQ_API_KEY)

            if page_items:
                all_items.extend(page_items)
            else:
                print(f"✗ No data extracted from page {idx}")

        final_json = {
            "restaurant_name": None,
            "phone": None,
            "categories": [
                {"category": "Menu", "items": all_items}
            ],
        }

        save_menu_json(final_json, menu_file)
