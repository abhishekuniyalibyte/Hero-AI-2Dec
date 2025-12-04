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


class MoodBasedMenuChatbot:
    def __init__(self, embeddings_path, model_name='sentence-transformers/all-mpnet-base-v2'):
        """
        Initialize the mood-based menu chatbot with embeddings and models.
        
        Args:
            embeddings_path: Path to the pickle file containing embeddings
            model_name: Sentence transformer model for encoding queries
        """
        print("Loading menu embeddings...")
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
            self.embeddings = data['embeddings']
            self.metadata = data['metadata']
        
        print(f"Loaded {len(self.embeddings)} menu items")
        
        print(f"Loading embedding model on CPU: {model_name}")
        self.encoder = SentenceTransformer(model_name, device="cpu")
        
        print("Initializing Groq client...")
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        
        # Conversation history
        self.conversation_history = []
        
        # Current detected mood
        self.current_mood = None
        
        # Mood-to-food preferences mapping
        self.mood_preferences = {
            'sad': ['comfort food', 'warm', 'hearty', 'creamy', 'chocolate', 'dessert', 'soup', 'pasta'],
            'happy': ['fresh', 'light', 'healthy', 'salad', 'grilled', 'colorful', 'fruit'],
            'excited': ['spicy', 'bold', 'flavorful', 'exotic', 'adventurous', 'tangy'],
            'celebration': ['premium', 'special', 'indulgent', 'deluxe', 'fancy', 'rich', 'feast'],
            'stressed': ['soothing', 'simple', 'familiar', 'easy', 'mild', 'tea', 'smoothie'],
            'tired': ['energizing', 'protein', 'nutritious', 'refreshing', 'coffee', 'juice'],
            'romantic': ['elegant', 'wine', 'intimate', 'special', 'dessert', 'sharing'],
            'nostalgic': ['traditional', 'classic', 'homestyle', 'authentic', 'comfort']
        }
        
        print("Chatbot ready!\n")
    
    def detect_mood(self, user_query):
        """
        Detect the user's mood from their message.
        
        Args:
            user_query: User's message
        
        Returns:
            Detected mood as string or None
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are a mood detection expert. Analyze the user's message and detect their emotional state.

Respond with ONLY ONE WORD from this list:
- sad (feeling down, upset, lonely, heartbroken)
- happy (cheerful, content, joyful, good mood)
- excited (energetic, enthusiastic, pumped)
- celebration (birthday, anniversary, achievement, party)
- stressed (anxious, overwhelmed, worried)
- tired (exhausted, drained, sleepy)
- romantic (date night, romantic dinner)
- nostalgic (missing home, childhood memories)
- neutral (no specific mood detected)

RESPOND WITH ONLY THE MOOD WORD, NOTHING ELSE."""
                },
                {
                    "role": "user",
                    "content": f"Detect the mood from this message: '{user_query}'"
                }
            ]
            
            completion = self.groq_client.chat.completions.create(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                messages=messages,
                temperature=0.3,
                max_tokens=10
            )
            
            mood = completion.choices[0].message.content.strip().lower()
            
            # Validate mood
            valid_moods = list(self.mood_preferences.keys()) + ['neutral']
            if mood in valid_moods and mood != 'neutral':
                return mood
            
            return None
        
        except Exception as e:
            print(f"Mood detection error: {e}")
            return None
    
    def enhance_query_with_mood(self, query, mood):
        """
        Enhance the search query with mood-relevant terms.
        
        Args:
            query: Original user query
            mood: Detected mood
        
        Returns:
            Enhanced query string
        """
        if mood and mood in self.mood_preferences:
            mood_terms = ' '.join(self.mood_preferences[mood][:3])
            return f"{query} {mood_terms}"
        return query
    
    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search_menu(self, query, mood=None, top_k=7):
        """
        Search for relevant menu items based on query and mood.
        
        Args:
            query: User's search query
            mood: User's mood (optional)
            top_k: Number of top results to return
        
        Returns:
            List of relevant menu items with metadata
        """
        # Enhance query with mood preferences
        enhanced_query = self.enhance_query_with_mood(query, mood)
        
        # Encode the query
        query_embedding = self.encoder.encode(enhanced_query)
        
        # Calculate similarities
        similarities = []
        for idx, embedding in enumerate(self.embeddings):
            sim = self.cosine_similarity(query_embedding, embedding)
            
            # Boost score if item matches mood preferences
            if mood and mood in self.mood_preferences:
                meta = self.metadata[idx]
                item_text = json.dumps(meta).lower()
                
                # Check if item description matches mood keywords
                mood_match_count = sum(1 for term in self.mood_preferences[mood] 
                                      if term.lower() in item_text)
                
                # Apply boost (up to 10% increase)
                mood_boost = min(0.1, mood_match_count * 0.02)
                sim += mood_boost
            
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
    
    def get_mood_system_prompt(self, mood):
        """
        Get a mood-appropriate system prompt.
        
        Args:
            mood: Detected mood
        
        Returns:
            System prompt string
        """
        mood_prompts = {
            'sad': """You are a warm, empathetic restaurant assistant. The customer is feeling down, so be extra caring and supportive. 
Recommend comfort foods that might lift their spirits. Use a gentle, understanding tone.""",
            
            'happy': """You are an enthusiastic and upbeat restaurant assistant. The customer is in a great mood! 
Match their energy with cheerful recommendations. Suggest fresh, vibrant dishes.""",
            
            'excited': """You are an energetic restaurant assistant. The customer is excited and looking for something special! 
Recommend bold, flavorful, or adventurous dishes. Be enthusiastic about your suggestions.""",
            
            'celebration': """You are a celebratory restaurant assistant. The customer is celebrating something special! 
Recommend premium, indulgent items perfect for the occasion. Make them feel special.""",
            
            'stressed': """You are a calming, reassuring restaurant assistant. The customer seems stressed. 
Recommend soothing, simple, familiar foods. Use a calm and reassuring tone.""",
            
            'tired': """You are an understanding restaurant assistant. The customer seems tired and needs energy. 
Recommend energizing, nutritious options. Be supportive and helpful.""",
            
            'romantic': """You are a sophisticated restaurant assistant. The customer is planning something romantic. 
Recommend elegant dishes perfect for sharing or special occasions. Be thoughtful and refined.""",
            
            'nostalgic': """You are a warm restaurant assistant. The customer is feeling nostalgic. 
Recommend traditional, homestyle, classic dishes. Connect with their memories."""
        }
        
        base_prompt = """You are a friendly and helpful restaurant menu assistant. Your role is to help customers find items from the menu and answer their questions.

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
        
        if mood and mood in mood_prompts:
            return mood_prompts[mood] + "\n\n" + base_prompt
        
        return base_prompt
    
    def generate_response(self, user_query, context, mood=None):
        """
        Generate conversational response using Groq API with mood awareness.
        
        Args:
            user_query: User's question
            context: Retrieved menu items context
            mood: Detected mood
        
        Returns:
            AI response string
        """
        # Build conversation with history
        messages = [
            {
                "role": "system",
                "content": self.get_mood_system_prompt(mood)
            }
        ]
        
        # Add conversation history (last 6 messages for context)
        for msg in self.conversation_history[-6:]:
            messages.append(msg)
        
        # Build user message with mood context
        user_content = f"""Customer question: {user_query}"""
        
        if mood:
            user_content += f"\n(Customer mood: {mood})"
        
        user_content += f"""

Available menu items:
{context}

Please answer the customer's question based ONLY on the menu items listed above. Be friendly and conversational."""
        
        if mood:
            user_content += f" Tailor your recommendations to their {mood} mood."
        
        messages.append({
            "role": "user",
            "content": user_content
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
        Main chat function with mood detection.
        
        Args:
            user_query: User's question/message
        
        Returns:
            AI response
        """
        # Detect mood
        detected_mood = self.detect_mood(user_query)
        
        if detected_mood:
            self.current_mood = detected_mood
            print(f"[Detected mood: {detected_mood}]")
        
        # Search for relevant menu items with mood enhancement
        search_results = self.search_menu(user_query, mood=self.current_mood, top_k=7)
        
        # Format context
        context = self.format_context(search_results)
        
        # Generate response with mood awareness
        response = self.generate_response(user_query, context, mood=self.current_mood)
        
        return response
    
    def reset_conversation(self):
        """Clear conversation history and mood."""
        self.conversation_history = []
        self.current_mood = None
        print("Conversation history and mood cleared.")


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
    chatbot = MoodBasedMenuChatbot(embeddings_path)
    
    print("=" * 60)
    print("MOOD-BASED RESTAURANT MENU CHATBOT")
    print("=" * 60)
    print("Ask me anything about our menu! I'll detect your mood")
    print("and recommend food accordingly.")
    print()
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