# chatbot to run on thenlper/gte-large

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")


class MenuChatbot:
    def __init__(self, embeddings_path, model_name='thenlper/gte-large'):
        """
        Initialize the menu chatbot with embeddings and models.
        
        Args:
            embeddings_path: Path to the pickle file containing embeddings
            model_name: Sentence transformer model for encoding queries (must match embedding generation)
        """
        print("Loading menu embeddings...")
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
            self.embeddings = data['embeddings']
            self.metadata = data['metadata']
        
        print(f"Loaded {len(self.embeddings)} menu items")
        
        print(f"Loading embedding model on CPU: {model_name}")
        # Must use the same model as embedding generation (thenlper/gte-large)
        self.encoder = SentenceTransformer(model_name, device="cpu")
        
        print("Initializing Groq client...")
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        
        # Conversation history
        self.conversation_history = []
        
        print("Chatbot ready!\n")
    
    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search_menu(self, query, top_k=5):
        """
        Search for relevant menu items based on query.
        
        Args:
            query: User's search query
            top_k: Number of top results to return
        
        Returns:
            List of relevant menu items with metadata
        """
        # Encode the query with normalization (to match embedding generation)
        query_embedding = self.encoder.encode(
            query,
            normalize_embeddings=True  # Must match embedding generation
        )
        
        # Calculate similarities
        similarities = []
        for idx, embedding in enumerate(self.embeddings):
            sim = self.cosine_similarity(query_embedding, embedding)
            similarities.append((idx, sim))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k results
        results = []
        for idx, sim in similarities[:top_k]:
            results.append({
                'metadata': self.metadata[idx],
                'similarity': float(sim)
            })
        
        return results
    
    def format_context(self, search_results):
        """Format search results into context for the LLM."""
        context_parts = []
        
        for idx, result in enumerate(search_results, 1):
            meta = result['metadata']
            item_info = []
            
            if meta.get('name'):
                item_info.append(f"Item {idx}: {meta['name']}")
            if meta.get('category'):
                item_info.append(f"Category: {meta['category']}")
            if meta.get('price'):
                item_info.append(f"Price: {meta['price']}")
            
            # Include original data if available
            if meta.get('original_data'):
                orig = meta['original_data']
                if orig.get('description'):
                    item_info.append(f"Description: {orig['description']}")
                if orig.get('ingredients'):
                    ing = orig['ingredients'] if isinstance(orig['ingredients'], str) else ', '.join(orig['ingredients'])
                    item_info.append(f"Ingredients: {ing}")
                if orig.get('allergens'):
                    all = orig['allergens'] if isinstance(orig['allergens'], str) else ', '.join(orig['allergens'])
                    item_info.append(f"Allergens: {all}")
                if orig.get('dietary_info'):
                    diet = orig['dietary_info'] if isinstance(orig['dietary_info'], str) else ', '.join(orig['dietary_info'])
                    item_info.append(f"Dietary: {diet}")
            
            context_parts.append('\n'.join(item_info))
        
        return '\n\n'.join(context_parts)
    
    def generate_response(self, user_query, context):
        """
        Generate conversational response using Groq API.
        
        Args:
            user_query: User's question
            context: Retrieved menu items context
        
        Returns:
            AI response string
        """
        # Build conversation with history
        messages = [
            {
                "role": "system",
                "content": """You are a friendly and helpful restaurant menu assistant. Your role is to help customers find items from the menu and answer their questions.

IMPORTANT RULES:
1. ONLY recommend items that are in the provided menu context below
2. DO NOT make up or suggest items that are not in the menu
3. If asked about something not in the menu, politely say it's not available
4. Be conversational, warm, and helpful
5. If asked about prices, ingredients, or details, provide accurate information from the menu
6. ALL PRICES ARE IN INR (Indian Rupees). Always mention prices as "₹X" or "Rs. X" or "INR X"
7. If the customer's question is unclear, ask for clarification
8. Make recommendations based on what's actually available in the menu
9. Keep responses concise but friendly"""
            }
        ]
        
        # Add conversation history (last 6 messages for context)
        for msg in self.conversation_history[-6:]:
            messages.append(msg)
        
        # Add current query with context
        messages.append({
            "role": "user",
            "content": f"""Customer question: {user_query}

Available menu items:
{context}

Please answer the customer's question based ONLY on the menu items listed above. Be friendly and conversational."""
        })
        
        try:
            completion = self.groq_client.chat.completions.create(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )
            
            response = completion.choices[0].message.content.strip()
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_query
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            
            return response
        
        except Exception as e:
            return f"I'm sorry, I encountered an error: {str(e)}"
    
    def chat(self, user_query):
        """
        Main chat function - handles user query and returns response.
        
        Args:
            user_query: User's question/message
        
        Returns:
            AI response
        """
        # Search for relevant menu items
        search_results = self.search_menu(user_query, top_k=5)
        
        # Format context
        context = self.format_context(search_results)
        
        # Generate response
        response = self.generate_response(user_query, context)
        
        return response
    
    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.")


def main():
    """Interactive chatbot interface."""
    import sys
    
    # Check if embeddings file path is provided
    if len(sys.argv) < 2:
        embeddings_path = "menu_embeddings.pkl"
        print(f"Using default embeddings file: {embeddings_path}")
    else:
        embeddings_path = sys.argv[1]
    
    if not os.path.exists(embeddings_path):
        print(f"Embeddings file not found: {embeddings_path}")
        print("Please provide the correct path to your menu_embeddings.pkl file")
        exit(1)
    
    # Initialize chatbot
    chatbot = MenuChatbot(embeddings_path)
    
    print("=" * 60)
    print("RESTAURANT MENU CHATBOT")
    print("=" * 60)
    print("Ask me anything about our menu!")
    print("Commands:")
    print("  - 'quit' or 'exit': Exit the chatbot")
    print("  - 'reset': Clear conversation history")
    print("=" * 60)
    print()
    
    # Interactive loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nThanks for chatting! Have a great day!")
                break
            
            if user_input.lower() == 'reset':
                chatbot.reset_conversation()
                continue
            
            # Get response
            print("\nAssistant: ", end="", flush=True)
            response = chatbot.chat(user_input)
            print(response)
            print()
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print()


if __name__ == "__main__":
    main()