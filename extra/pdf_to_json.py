"""
Simple PDF/Image Text Extraction Bot
Extracts text from PDFs and JPEG images, creates JSON output using Groq LLM.
"""
import os
import json
import PyPDF2
from pathlib import Path
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
from PIL import Image
import pytesseract

# Load environment variables
load_dotenv()


def extract_text_from_image(image_path):
    """Extract text from JPEG/PNG image using OCR."""
    try:
        # Open image
        image = Image.open(image_path)
        
        # Perform OCR
        text = pytesseract.image_to_string(image)
        
        if text.strip():
            print(f"Extracted text from image using OCR")
            return text, 1  # Images count as 1 page
        else:
            print("Warning: No text found in image")
            return None, 0
    
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return None, 0


def preprocess_menu_text(text):
    """Clean and preprocess extracted menu text for better LLM processing."""
    import re
    
    cleaned_text = text
    
    # Remove phone numbers (common pattern in menus)
    # Matches patterns like: +91 1234567890, 123-456-7890, (123) 456-7890
    cleaned_text = re.sub(r'[\+\(]?\d{1,4}[\)\-\s]?\d{3,4}[\-\s]?\d{3,4}[\-\s]?\d{3,4}', '', cleaned_text)
    
    # Remove email addresses
    cleaned_text = re.sub(r'\S+@\S+\.\S+', '', cleaned_text)
    
    # Remove URLs
    cleaned_text = re.sub(r'http[s]?://\S+|www\.\S+', '', cleaned_text)
    
    # Remove extra whitespace while preserving structure
    cleaned_text = re.sub(r' +', ' ', cleaned_text)  # Multiple spaces to single space
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)  # Multiple newlines to double newline
    
    return cleaned_text.strip()


def extract_text_from_pdf(pdf_path):
    """Extract text from all pages of a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)

            all_text = []
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                all_text.append(text)

            print(f"Extracted text from {total_pages} pages")
            return "\n\n--- PAGE BREAK ---\n\n".join(all_text), total_pages

    except Exception as e:
        print(f"Error extracting text: {e}")
        return None, 0


def process_with_llm(text, filename):
    """Use Groq LLM to structure the extracted text into JSON."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    # Load config from .env
    model = os.getenv("MODEL_NAME")
    temperature = float(os.getenv("TEMPERATURE", 0.3))
    max_tokens = int(os.getenv("MAX_TOKENS", 4000))

    client = Groq(api_key=api_key)

    # Detect menu-type content
    is_menu = any(keyword in text.lower() for keyword in [
        "menu", "appetizer", "entree", "dessert", "price", "$", "₹"
    ])

    if is_menu:
        system_prompt = """You are an expert at analyzing restaurant menus. Extract structured information accurately.

CRITICAL INSTRUCTIONS FOR MENU PARSING:

1. **Category Detection**:
   - Category headers are typically standalone words/phrases like "South Indian", "Chinese", "Beverages", "Signature Snacks", "Healthy Option"
   - Items that appear BEFORE the first category header should be assigned category "Breakfast" or "Snacks"
   - Items that appear AFTER a category header belong to that category UNTIL the next category header
   - Example: If you see "VEG BURGER" then "South Indian" then "IDLI", VEG BURGER is NOT South Indian

2. **Price Extraction**:
   - Prices are typically numbers (with or without currency symbols ₹, $)
   - In menu layouts, prices often appear on the same line or the next line after the item name
   - Look for number patterns: 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 180, etc.
   - If an item name is followed by a number, that number is likely the price
   - Extract EXACT prices as shown, preserve currency symbols if present

3. **Item Name Extraction**:
   - Item names are typically in ALL CAPS
   - Keep item names EXACTLY as shown
   - Common items: BURGER, PARATHA, MAGGI, SANDWICH, DOSA, IDLI, NOODLES, TEA, COFFEE, SHAKE

4. **Layout Understanding**:
   - Menu text may have items and prices on different lines due to PDF extraction
   - Use context and number patterns to match items with their prices
   - If you see a sequence like "ITEM_NAME 60", the price is 60

5. **Validation**:
   - Every menu item should have a price (very rare to have null)
   - If you can't find a price, look at nearby numbers
   - Category should never be null if category headers are present

Return ONLY valid JSON in this format:
{
  "restaurant_name": "string or null",
  "items": [
    {
      "name": "EXACT item name as shown",
      "description": "description or null",
      "price": "price with currency symbol (e.g., ₹60) or null only if truly not visible",
      "category": "category from section header or 'Breakfast' for items before first header"
    }
  ]
}

EXAMPLES:
- Text: "VEG BURGER 60" → {"name": "VEG BURGER", "price": "₹60", "category": "Breakfast"}
- Text: "South Indian\nIDLI SAMBHAR 45" → {"name": "IDLI SAMBHAR", "price": "₹45", "category": "South Indian"}
- Text: "MASALA TEA 20" under "Beverages" → {"name": "MASALA TEA", "price": "₹20", "category": "Beverages"}"""
        print("Processing as restaurant menu...")
    else:
        system_prompt = """You are an expert at analyzing documents. Extract key information and create a summary.

Return ONLY valid JSON in this format:
{
  "title": "document title or null",
  "summary": "brief summary of the document",
  "key_points": ["point 1", "point 2"]
}"""
        print("Processing as general document...")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract structured information from this text:\n\n{text}"}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content.strip()

        # Extract JSON from fenced blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        structured_data = json.loads(content)
        print("Successfully processed with LLM")
        return structured_data

    except Exception as e:
        print(f"Error in LLM processing: {e}")
        return {"error": str(e), "raw_text": text[:500]}


def save_json(data, output_path):
    """Save data as JSON file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving JSON: {e}")
        return False


def main():
    """Automatically process files from data/ directory (PDF or images)."""
    print("\n" + "=" * 50)
    print("PDF/Image Text Extraction Bot")
    print("=" * 50 + "\n")

    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    output_dir = Path(os.getenv("OUTPUT_DIR", script_dir / "output"))

    # Look for menu files (PDF or images)
    supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png']
    menu_files = []
    
    for ext in supported_extensions:
        menu_files.extend(list(data_dir.glob(f"menu{ext}")))
    
    if not menu_files:
        print(f"Error: No menu file found in {data_dir}")
        print(f"Looking for: menu.pdf, menu.jpg, menu.jpeg, or menu.png")
        return
    
    # Process the first menu file found
    file_path = menu_files[0]
    print(f"Processing: {file_path.name}")

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Extract text based on file type
    file_ext = file_path.suffix.lower()
    
    if file_ext == '.pdf':
        text, total_pages = extract_text_from_pdf(file_path)
    elif file_ext in ['.jpg', '.jpeg', '.png']:
        text, total_pages = extract_text_from_image(file_path)
    else:
        print(f"Error: Unsupported file type: {file_ext}")
        return
    
    if not text:
        print("Failed to extract text")
        return

    # Preprocess text to fix OCR errors
    print("Preprocessing text...")
    text = preprocess_menu_text(text)

    # Process using LLM
    structured_data = process_with_llm(text, file_path.name)

    # Final output JSON payload
    output_data = {
        "metadata": {
            "filename": file_path.name,
            "total_pages": total_pages,
            "processing_date": datetime.now().isoformat(),
            "model_used": os.getenv("MODEL_NAME"),
            "file_type": file_ext[1:],  # Remove the dot
        },
        "data": structured_data,
    }

    # Output JSON file
    output_path = output_dir / f"{file_path.stem}.json"

    if save_json(output_data, output_path):
        print("\n" + "=" * 50)
        print("Processing complete")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
