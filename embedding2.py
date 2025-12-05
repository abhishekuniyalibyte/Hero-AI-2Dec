# Hero AI: embedding using thenlper/gte-large

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
from typing import List, Dict, Any
import argparse

class MenuEmbeddingGenerator:
    def __init__(self, model_name: str = "thenlper/gte-large"):
        """
        Initialize the embedding generator with a sentence transformer model.
        Force CPU usage only.
        """
        print(f"Loading model on CPU: {model_name}")
        self.model = SentenceTransformer(model_name, device="cpu")
        self.embeddings = []
        self.metadata = []
    
    def load_menu_json(self, json_path: str) -> Dict[str, Any]:
        """Load menu data from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_text_chunks(self, menu_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert menu JSON into text chunks suitable for embedding."""
        chunks = []
        
        items = []
        if isinstance(menu_data, dict):
            if 'categories' in menu_data:
                for category in menu_data['categories']:
                    cat_name = category.get('name', 'Unknown')
                    for item in category.get('items', []):
                        item['category'] = cat_name
                        items.append(item)
            elif 'items' in menu_data:
                items = menu_data['items']
            else:
                items = menu_data.get('menu', [])
        elif isinstance(menu_data, list):
            items = menu_data
        
        for idx, item in enumerate(items):
            text_parts = []
            
            name = item.get('name', item.get('item_name', ''))
            if name:
                text_parts.append(f"Item: {name}")

            category = item.get('category', item.get('type', ''))
            if category:
                text_parts.append(f"Category: {category}")

            description = item.get('description', item.get('desc', ''))
            if description:
                text_parts.append(f"Description: {description}")

            price = item.get('price', item.get('cost', ''))
            if price:
                text_parts.append(f"Price: {price}")

            ingredients = item.get('ingredients', [])
            if ingredients:
                ing_str = ', '.join(ingredients) if isinstance(ingredients, list) else ingredients
                text_parts.append(f"Ingredients: {ing_str}")

            allergens = item.get('allergens', [])
            if allergens:
                all_str = ', '.join(allergens) if isinstance(allergens, list) else allergens
                text_parts.append(f"Allergens: {all_str}")

            dietary = item.get('dietary_info', [])
            if dietary:
                diet_str = ', '.join(dietary) if isinstance(dietary, list) else dietary
                text_parts.append(f"Dietary: {diet_str}")

            text = ". ".join(text_parts)

            chunks.append({
                'text': text,
                'metadata': {
                    'item_id': idx,
                    'name': name,
                    'category': category,
                    'price': price,
                    'original_data': item
                }
            })
        
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> None:
        """Generate embeddings for all text chunks."""
        print(f"Generating embeddings for {len(chunks)} items on CPU...")
        texts = [chunk['text'] for chunk in chunks]

        # gte-large works normally with encode()
        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True  # recommended for gte models
        )

        self.metadata = [chunk['metadata'] for chunk in chunks]
        print(f"Generated {len(self.embeddings)} embeddings")
    
    def save_embeddings(self, output_path: str, format: str = 'pickle') -> None:
        """Save embeddings + metadata to file."""
        output_path = Path(output_path)
        
        if format == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump({
                    'embeddings': self.embeddings,
                    'metadata': self.metadata
                }, f)
        
        elif format == 'npz':
            np.savez(output_path,
                     embeddings=self.embeddings,
                     metadata=np.array(self.metadata, dtype=object))
        
        elif format == 'json':
            data = {
                'embeddings': self.embeddings.tolist(),
                'metadata': self.metadata
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        
        print(f"Saved embeddings to {output_path}")
    
    def process_menu(self, json_path: str, output_path: str, format: str = 'pickle') -> None:
        menu_data = self.load_menu_json(json_path)
        chunks = self.create_text_chunks(menu_data)
        self.generate_embeddings(chunks)
        self.save_embeddings(output_path, format)


def main():
    parser = argparse.ArgumentParser(description='Generate embeddings from restaurant menu JSON')
    parser.add_argument('input_json', help='Path to input menu JSON file')
    parser.add_argument('output_file', help='Path to output embeddings file')
    parser.add_argument('--format', choices=['pickle', 'npz', 'json'],
                       default='pickle', help='Output format (default: pickle)')
    parser.add_argument('--model', default="thenlper/gte-large",
                       help='Sentence transformer model (default: thenlper/gte-large)')
    
    args = parser.parse_args()
    
    generator = MenuEmbeddingGenerator(model_name=args.model)
    generator.process_menu(args.input_json, args.output_file, args.format)
    
    print("\nEmbedding generation complete!")
    print(f"  Input: {args.input_json}")
    print(f"  Output: {args.output_file}")
    print(f"  Format: {args.format}")
    print(f"  Model: {args.model}")
    print(f"  Total items: {len(generator.metadata)}")


if __name__ == "__main__":
    main()
