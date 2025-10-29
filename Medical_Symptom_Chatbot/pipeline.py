import os
import json
import numpy as np
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MedicalChatbot:
    def __init__(self, data_path="data"):
        self.data_path = data_path
        self.data = self.load_json_data()
        self.index = self.build_semantic_index()
        
        # Configure Gemini with API key from environment
        GEMINI_API_KEY = "AIzaSyAyGz15x3nmH3gl_VF7vD8SLkcbyhms5L0"
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Models
        self.gen_model = genai.GenerativeModel("models/gemini-2.0-flash-lite")
        self.embed_model = "models/text-embedding-004"

    # -------------------------------
    # LOAD JSON DATA FROM SUBFOLDERS
    # -------------------------------
    def load_json_data(self):
        data = {}
        try:
            base_path = Path(self.data_path)
            
            # Define the folder structure
            folders = {
                "conditions": ["common_conditions"],
                "symptoms": ["common_symptoms","emergency_symptoms"],
                "medications": ["drug_database"],
                "general_health": ["prevention_guidelines"],
                "first_aid": ["emergency_procedure"]
            }
            
            for folder, files in folders.items():
                data[folder] = {}
                for file_name in files:
                    file_path = base_path / folder / f"{file_name}.json"
                    if file_path.exists():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data[folder][file_name] = json.load(f)
                    else:
                        print(f"Warning: File not found: {file_path}")
            
            return data
            
        except Exception as e:
            print(f"Error loading JSON data: {e}")
            return {}

    # -------------------------------
    # BUILD SEMANTIC INDEX
    # -------------------------------
    def build_semantic_index(self):
        index = []
        try:
            for domain, files in self.data.items():
                for file_name, content in files.items():
                    # Convert JSON content to text for embedding
                    text = self.json_to_text(content, f"{domain}/{file_name}")
                    if text:
                        embedding_data = genai.embed_content(model=self.embed_model, content=text)
                        embedding = embedding_data["embedding"]
                        index.append({
                            "domain": domain,
                            "file": file_name,
                            "text": text,
                            "embedding": embedding,
                            "original_data": content
                        })
            return index
        except Exception as e:
            print(f"Error building index: {e}")
            return []

    # -------------------------------
    # CONVERT JSON TO SEARCHABLE TEXT
    # -------------------------------
    def json_to_text(self, json_data, source):
        try:
            if isinstance(json_data, dict):
                text_parts = []
                for key, value in json_data.items():
                    if isinstance(value, dict):
                        text_parts.append(f"{key}: {self.json_to_text(value, '')}")
                    elif isinstance(value, list):
                        text_parts.append(f"{key}: {'; '.join(map(str, value))}")
                    else:
                        text_parts.append(f"{key}: {value}")
                return f"[{source}] " + " | ".join(text_parts)
            elif isinstance(json_data, list):
                return f"[{source}] " + "; ".join(map(str, json_data))
            else:
                return f"[{source}] {str(json_data)}"
        except Exception as e:
            return str(json_data)

    # -------------------------------
    # SEMANTIC SEARCH
    # -------------------------------
    def semantic_search(self, query, top_k=5):
        try:
            if not self.index:
                return []
                
            q_embed = genai.embed_content(model=self.embed_model, content=query)["embedding"]
            scored = []
            for item in self.index:
                score = np.dot(q_embed, item["embedding"]) / (
                    np.linalg.norm(q_embed) * np.linalg.norm(item["embedding"])
                )
                scored.append((score, item))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [s[1] for s in scored[:top_k]]
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []

    # -------------------------------
    # CHECK IF QUERY IS IN DATABASE
    # -------------------------------
    def is_in_database(self, query):
        """Check if the query matches something in our database"""
        results = self.semantic_search(query, top_k=1)
        if results and results[0]['score'] > 0.3:  # Threshold for good match
            return True
        return False

    # -------------------------------
    # GENERATE INDEPENDENT MEDICAL RESPONSE
    # -------------------------------
    def generate_response(self, query):
        try:
            # Search medical context first
            results = self.semantic_search(query)
            
            # Determine if we have good database matches
            database_has_info = len(results) > 0 and any(result.get('score', 0) > 0.3 for result in results)
            
            if database_has_info:
                context = "\n\n".join([f"- From {r['domain']}/{r['file']}:\n{r['text']}" for r in results])
                context_note = "I found relevant information in our medical database:"
                source = "database"
            else:
                context = "No specific matching information found in our medical database for this query."
                context_note = "While this specific information isn't in our database, here's what I can tell you based on general medical knowledge:"
                source = "general_knowledge"

            # Enhanced prompt that works independently
            prompt = f"""
You are a medical information assistant. The user has asked: "{query}"

{context_note}
{context}

IMPORTANT SAFETY RULES (ALWAYS FOLLOW THESE):
1. For emergency situations (choking, chest pain, difficulty breathing, severe allergic reactions), IMMEDIATELY advise calling emergency services
2. Always include clear disclaimers that this is not medical advice
3. Strongly recommend consulting healthcare professionals for personal medical concerns
4. If discussing medications, emphasize consulting a doctor before taking any medication
5. Be accurate, conservative, and safety-focused in all information

RESPONSE GUIDELINES:
- If using database information, prioritize it but supplement with general knowledge when helpful
- If no database information, provide accurate general medical information with clear safety warnings
- Structure information clearly with headings and bullet points
- Include relevant categories: description, uses, dosage guidelines, side effects, precautions, when to seek help
- For medications: include common brand names, drug class, typical uses, important precautions
- Always end with a strong recommendation to consult healthcare providers

Provide a comprehensive, safety-focused response:
"""
            response = self.gen_model.generate_content(prompt)
            response_text = response.text.strip() if response and response.text else "I'm sorry, I couldn't generate a response."
            
            # Add source indicator
            if source == "database":
                response_text += "\n\n📚 *Information sourced from medical database*"
            else:
                response_text += "\n\n💡 *General medical information - consult healthcare provider for personalized advice*"
                
            return response_text

        except Exception as e:
            return f"⚠️ Error generating response: {e}"

    # -------------------------------
    # GET MEDICATION INFORMATION (INDEPENDENT)
    # -------------------------------
    def get_medication_info(self, drug_name):
        """Get comprehensive medication information independently"""
        try:
            prompt = f"""
Provide comprehensive, accurate information about the medication/drug: {drug_name}

Please structure your response with these sections:

**💊 {drug_name.upper()} - Medication Information**

**Drug Class & Common Brands:**
- [Drug class and common brand names]

**Primary Uses:**
- [Main medical uses and indications]

**Typical Dosage Guidelines:**
- [General adult dosage information]
- [Important dosage precautions]

**Side Effects:**
- Common: [list common side effects]
- Serious: [list serious side effects requiring medical attention]

**Important Precautions & Warnings:**
- [Contraindications and who should avoid this medication]
- [Drug interactions to be aware of]
- [Special populations (elderly, pregnant, children)]

**Key Safety Information:**
- When to avoid this medication
- When to seek immediate medical help
- Storage and administration instructions

CRITICAL SAFETY NOTES:
- ALWAYS emphasize that this is general information only
- STRONGLY recommend consulting a doctor or pharmacist before taking any medication
- Dosage must be determined by a healthcare professional
- Individual responses to medications vary

Provide accurate, conservative medical information:
"""
            response = self.gen_model.generate_content(prompt)
            return response.text.strip() if response and response.text else f"Could not retrieve information about {drug_name}."
        
        except Exception as e:
            return f"Error retrieving medication information: {e}"

    # -------------------------------
    # GET CONDITION INFORMATION (INDEPENDENT)
    # -------------------------------
    def get_condition_info(self, condition_name):
        """Get comprehensive condition information independently"""
        try:
            prompt = f"""
Provide comprehensive, accurate information about the medical condition: {condition_name}

Please structure your response with these sections:

**🩺 {condition_name.upper()} - Condition Overview**

**Description & Causes:**
- [Brief description and common causes]

**Common Symptoms:**
- [Main symptoms and presentation]

**Diagnosis & Treatment:**
- [How it's typically diagnosed]
- [Common treatment approaches]

**Self-Care & Management:**
- [Home care and management tips]
- [Lifestyle recommendations]

**When to Seek Medical Help:**
- [Red flags and emergency symptoms]
- [When to consult a healthcare provider]

**Prevention:**
- [Prevention strategies if applicable]

CRITICAL SAFETY NOTES:
- ALWAYS emphasize that this is general information only
- STRONGLY recommend consulting a healthcare professional for diagnosis and treatment
- Highlight emergency symptoms that require immediate medical attention

Provide accurate, helpful medical information:
"""
            response = self.gen_model.generate_content(prompt)
            return response.text.strip() if response and response.text else f"Could not retrieve information about {condition_name}."
        
        except Exception as e:
            return f"Error retrieving condition information: {e}"

    # -------------------------------
    # GET AVAILABLE TOPICS
    # -------------------------------
    def get_available_topics(self):
        topics = []
        for domain, files in self.data.items():
            for file_name in files.keys():
                topics.append(f"{domain}/{file_name}")
        return topics

    # -------------------------------
    # DETECT QUERY TYPE
    # -------------------------------
    def detect_query_type(self, query):
        """Detect what type of medical information is being requested"""
        query_lower = query.lower()
        
        # Medication-related keywords
        med_keywords = ['medicine', 'medication', 'drug', 'pill', 'tablet', 'capsule', 'dose', 'dosage', 
                       'side effect', 'take', 'prescription', 'ibuprofen', 'aspirin', 'paracetamol',
                       'antibiotic', 'antihistamine', 'statins', 'blood pressure medicine']
        
        # Condition-related keywords  
        condition_keywords = ['symptom', 'condition', 'disease', 'illness', 'sick', 'diagnosis',
                             'treatment', 'cure', 'what is', 'have', 'suffering from']
        
        # Emergency keywords
        emergency_keywords = ['emergency', 'urgent', 'immediate', 'right now', '911', '999',
                             'ambulance', 'ER', 'emergency room', 'critical']
        
        if any(keyword in query_lower for keyword in med_keywords):
            return "medication"
        elif any(keyword in query_lower for keyword in condition_keywords):
            return "condition"
        elif any(keyword in query_lower for keyword in emergency_keywords):
            return "emergency"
        else:
            return "general"