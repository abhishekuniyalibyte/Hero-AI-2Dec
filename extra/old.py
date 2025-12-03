import base64
import json
import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
 
# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
 
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")
 
def convert_pdf_to_image(pdf_path):
    try:
        from pdf2image import convert_from_path
        print("Converting PDF to image...")
        images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=300)
 
        pdf_dir = os.path.dirname(pdf_path) or "."
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        image_path = os.path.join(pdf_dir, f"{pdf_name}_converted.png")
 
        images[0].save(image_path, 'PNG')
        print(f"Saved as: {image_path}")
        return image_path
 
    except ImportError:
        print("pdf2image not installed. Run: pip install pdf2image")
        return None
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return None
 
 
def extract_menu_to_json(file_path, groq_api_key):
    client = Groq(api_key=groq_api_key)
 
    with open(file_path, "rb") as file:
        file_data = base64.b64encode(file.read()).decode("utf-8")
 
    prompt = """Extract all menu items from this restaurant menu image and return ONLY a valid JSON object.
 
Structure the JSON like this:
{
  "restaurant_name": "string or null",
  "phone": "string or null",
  "categories": [
    {
      "category": "string",
      "items": [
        {"name": "string", "price": number}
      ]
    }
  ]
}
 
Rules:
1. Extract all items with exact names and prices.
2. Group items by category headers.
3. Prices must be numbers only.
4. No explanations. Only JSON.
"""
 
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{file_data}"}
                        }
                    ]
                }
            ],
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.1,
            max_tokens=4096
        )
 
        response_text = chat_completion.choices[0].message.content.strip()
 
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
 
        menu_data = json.loads(response_text)
        return menu_data
 
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        print("Raw Response:")
        print(response_text)
        return None
    except Exception as e:
        print(f"API Error: {e}")
        return None
 
 
def save_menu_json(menu_data, input_path, output_filename=None):
    input_dir = os.path.dirname(input_path) or "."
 
    if output_filename is None:
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{input_name}_extracted.json"
 
    output_path = os.path.join(input_dir, output_filename)
 
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(menu_data, f, indent=2, ensure_ascii=False)
 
    print(f"Saved JSON to {output_path}")
    return output_path
 
 
if __name__ == "__main__":
    menu_file = "menu.pdf"
 
    if not os.path.exists(menu_file):
        print(f"File not found: {menu_file}")
        exit(1)
 
    print(f"Found: {menu_file}")
 
    if menu_file.lower().endswith(".pdf"):
        FILE_PATH = convert_pdf_to_image(menu_file)
        if not FILE_PATH:
            exit(1)
    else:
        FILE_PATH = menu_file
 
    print("Extracting menu data...")
 
    menu_data = extract_menu_to_json(FILE_PATH, GROQ_API_KEY)
 
    if menu_data:
        print("Extracted Menu Data:")
        print(json.dumps(menu_data, indent=2, ensure_ascii=False))
 
        save_menu_json(menu_data, menu_file)
 
        total_items = sum(len(cat["items"]) for cat in menu_data.get("categories", []))
        print("Summary:")
        print(f"Categories: {len(menu_data.get('categories', []))}")
        print(f"Total Items: {total_items}")
    else:
        print("Failed to extract menu data.")
 