# PDF/Image to JSON Bot

Extract text from restaurant menus (PDF or JPEG images) and convert to structured JSON using Groq AI.

## Features
- ✅ Extract from **PDF files**
- ✅ Extract from **JPEG/PNG images** (using OCR)
- ✅ Auto-detect file type
- ✅ Accurate price extraction with currency symbols
- ✅ Category detection from section headers
- ✅ Powered by Groq's Llama 4 Maverick model

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR (required for image processing)
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

## Setup

1. Add your Groq API key to `.env` file:
```bash
GROQ_API_KEY=your_api_key_here
MODEL_NAME=meta-llama/llama-4-maverick-17b-128e-instruct
OUTPUT_DIR=output
TEMPERATURE=0.3
MAX_TOKENS=4000
```

2. Place your menu file in the `data/` folder:
   - `data/menu.pdf` OR
   - `data/menu.jpg` OR
   - `data/menu.jpeg` OR
   - `data/menu.png`

## Usage

```bash
python pdf_to_json.py
```

The script will:
1. Auto-detect the menu file in `data/` folder
2. Extract text (using PDF reader or OCR)
3. Process with Groq AI to structure the data
4. Save JSON to `output/menu.json`

## Output Format

```json
{
  "metadata": {
    "filename": "menu.pdf",
    "total_pages": 1,
    "processing_date": "2025-12-02T17:00:00",
    "model_used": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "file_type": "pdf"
  },
  "data": {
    "restaurant_name": "CAFE UNO",
    "items": [
      {
        "name": "MASALA DOSA",
        "description": null,
        "price": "₹60",
        "category": "South Indian"
      }
    ]
  }
}
```

## Improvements Made

### Better Price Extraction
- Extracts exact prices as shown in menu
- Preserves currency symbols (₹, $, etc.)
- Won't make up prices if not visible

### Better Category Detection
- Identifies section headers (South Indian, Chinese, Beverages, etc.)
- Assigns items to correct categories
- Improved prompt instructions for accuracy

### Image Support
- Added OCR capability using Tesseract
- Supports JPEG, JPG, and PNG formats
- Automatically detects file type

## Troubleshooting

**"No menu file found"**
- Make sure you have a file named `menu.pdf`, `menu.jpg`, `menu.jpeg`, or `menu.png` in the `data/` folder

**"tesseract is not installed"**
- Install Tesseract OCR (see Installation section above)

**Wrong prices or categories**
- Check if the PDF/image text is clear and readable
- Try adjusting TEMPERATURE in .env (lower = more consistent)
- Verify the menu has clear section headers for categories
