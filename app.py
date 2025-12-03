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
        print("Converting PDF to images...")
        images = convert_from_path(pdf_path, dpi=300)
 
        pdf_dir = os.path.dirname(pdf_path) or "."
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        image_paths = []
        for i, img in enumerate(images, 1):
            image_path = os.path.join(pdf_dir, f"{pdf_name}_page{i}.png")
            img.save(image_path, 'PNG')
            image_paths.append(image_path)
        
        print(f"Saved {len(image_paths)} page(s)")
        return image_paths
 
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
3. If an item has multiple prices (different sizes, variants, or options), create separate entries for each.
4. Include size/variant information in the item name to distinguish them.
5. Prices must be numbers only (no strings like "180/190").
6. No explanations. Only JSON.
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
            #max_tokens=8192
        )
 
        response_text = chat_completion.choices[0].message.content.strip()
 
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
 
        menu_data = json.loads(response_text)
        return menu_data
 
    except json.JSONDecodeError as e:
        print(f"⚠ JSON Parse Error: {e}")
        print(f"⚠ Attempting to fix malformed JSON...")
        
        # Try to salvage partial JSON
        try:
            last_brace = response_text.rfind('}')
            if last_brace > 0:
                test_json = response_text[:last_brace+1]
                open_braces = test_json.count('{')
                close_braces = test_json.count('}')
                test_json += '}' * (open_braces - close_braces)
                
                menu_data = json.loads(test_json)
                print("✓ Successfully recovered partial data")
                return menu_data
        except:
            pass
            
        print("✗ Could not recover data from this page")
        print("Raw Response (first 500 chars):")
        print(response_text[:500])
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


def cleanup_images(image_paths):
    """Delete temporary image files created from PDF conversion"""
    for path in image_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"Warning: Could not delete {path}: {e}")
 
 
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 script.py <menu_file>")
        print("Supported formats: PDF, JPG, JPEG, PNG")
        exit(1)
    
    menu_file = sys.argv[1]
 
    if not os.path.exists(menu_file):
        print(f"File not found: {menu_file}")
        exit(1)
 
    print(f"Found: {menu_file}")
 
    is_pdf = menu_file.lower().endswith(".pdf")
    
    if is_pdf:
        FILE_PATHS = convert_pdf_to_image(menu_file)
        if not FILE_PATHS:
            exit(1)
    else:
        FILE_PATHS = [menu_file]
 
    print("Extracting menu data...")
    
    all_categories = []
    restaurant_name = None
    phone = None
 
    for i, path in enumerate(FILE_PATHS, 1):
        print(f"\n{'='*50}")
        print(f"Processing page {i}/{len(FILE_PATHS)}...")
        print(f"{'='*50}")
        menu_data = extract_menu_to_json(path, GROQ_API_KEY)
        
        if menu_data:
            if not restaurant_name:
                restaurant_name = menu_data.get("restaurant_name")
            if not phone:
                phone = menu_data.get("phone")
            all_categories.extend(menu_data.get("categories", []))
            print(f"✓ Successfully extracted {len(menu_data.get('categories', []))} categories from page {i}")
        else:
            print(f"✗ Failed to extract data from page {i}")
    
    # Clean up temporary images if PDF was converted
    if is_pdf:
        print("\nCleaning up temporary image files...")
        cleanup_images(FILE_PATHS)
        print("✓ Temporary files removed")
    
    combined_menu = {
        "restaurant_name": restaurant_name,
        "phone": phone,
        "categories": all_categories
    }
 
    if combined_menu["categories"]:
        print("\n" + "="*50)
        print("EXTRACTION COMPLETE")
        print("="*50)
 
        save_menu_json(combined_menu, menu_file)
 
        total_items = sum(len(cat["items"]) for cat in combined_menu.get("categories", []))
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Pages Processed: {len(FILE_PATHS)}")
        print(f"Categories Extracted: {len(combined_menu.get('categories', []))}")
        print(f"Total Items: {total_items}")
    else:
        print("\n✗ No data extracted from any page!")