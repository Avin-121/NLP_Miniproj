import os
import pandas as pd
import numpy as np
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

genai.configure(api_key=GEMINI_API_KEY)

# Models
GEN_MODEL = genai.GenerativeModel("models/gemini-2.0-flash-lite")
EMBED_MODEL = "models/text-embedding-004"


class MedicalChatbot:
    def __init__(self, data_path="data"):
        self.data_path = data_path
        self.data = self.load_csv_data()
        self.index = self.build_semantic_index()

    # -------------------------------
    # LOAD CSV DATA
    # -------------------------------
    def load_csv_data(self):
        data = {}
        try:
            base_path = Path(self.data_path)
            
            # Load conditions data
            conditions_path = base_path / "conditions.csv"
            if conditions_path.exists():
                data['conditions'] = pd.read_csv(conditions_path)
            else:
                print(f"Warning: {conditions_path} not found")
            
            # Load medications data
            drugs_path = base_path / "drugs.csv"
            if drugs_path.exists():
                data['drugs'] = pd.read_csv(drugs_path)
            else:
                print(f"Warning: {drugs_path} not found")
            
            # Load symptoms data
            symptoms_path = base_path / "symptoms.csv"
            if symptoms_path.exists():
                data['symptoms'] = pd.read_csv(symptoms_path)
            else:
                print(f"Warning: {symptoms_path} not found")
            
            # Load solutions data
            solutions_path = base_path / "solutions.csv"
            if solutions_path.exists():
                data['solutions'] = pd.read_csv(solutions_path)
            else:
                print(f"Warning: {solutions_path} not found")
            
            return data
            
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            return {}

    # -------------------------------
    # BUILD SEMANTIC INDEX FROM CSV
    # -------------------------------
    def build_semantic_index(self):
        index = []
        try:
            for data_type, df in self.data.items():
                if isinstance(df, pd.DataFrame):
                    for _, row in df.iterrows():
                        # Convert row to searchable text
                        text = self.row_to_text(row, data_type)
                        if text:
                            embedding_data = genai.embed_content(model=EMBED_MODEL, content=text)
                            embedding = embedding_data["embedding"]
                            index.append({
                                "type": data_type,
                                "data": row.to_dict(),
                                "text": text,
                                "embedding": embedding
                            })
            return index
        except Exception as e:
            print(f"Error building index: {e}")
            return []

    # -------------------------------
    # CONVERT CSV ROW TO SEARCHABLE TEXT
    # -------------------------------
    def row_to_text(self, row, data_type):
        try:
            if data_type == 'conditions':
                return f"Condition: {row.get('name', '')} | Symptoms: {row.get('symptoms', '')} | Treatment: {row.get('treatment', '')} | Prevention: {row.get('prevention', '')}"
            
            elif data_type == 'drugs':
                return f"Drug: {row.get('drug_name', '')} | Class: {row.get('drug_class', '')} | Uses: {row.get('uses', '')} | Side Effects: {row.get('side_effects', '')} | Contraindications: {row.get('contraindications', '')}"
            
            elif data_type == 'symptoms':
                return f"Symptom: {row.get('name', '')} | Possible Conditions: {row.get('possible_conditions', '')} | Severity: {row.get('severity', '')} | When to See Doctor: {row.get('when_to_see_doctor', '')}"
            
            elif data_type == 'solutions':
                return f"Problem: {row.get('problem', '')} | Solution: {row.get('solution', '')} | Steps: {row.get('steps', '')} | Precautions: {row.get('precautions', '')}"
            
            return str(row.to_dict())
        except Exception as e:
            return str(row.to_dict())

    # -------------------------------
    # SEMANTIC SEARCH
    # -------------------------------
    def semantic_search(self, query, top_k=5):
        try:
            if not self.index:
                return []
                
            q_embed = genai.embed_content(model=EMBED_MODEL, content=query)["embedding"]
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
    # GET SPECIFIC DATA FROM CSV
    # -------------------------------
    def get_condition_info(self, condition_name):
        """Get condition information from CSV data"""
        if 'conditions' in self.data:
            df = self.data['conditions']
            condition_lower = condition_name.lower()
            match = df[df['name'].str.lower().str.contains(condition_lower, na=False)]
            if not match.empty:
                row = match.iloc[0]
                return self.format_condition_response(row)
        
        # Fallback to Gemini if not found in CSV
        return self.generate_condition_info(condition_name)

    def get_drug_info(self, drug_name):
        """Get drug information from CSV data with new column structure"""
        if 'drugs' in self.data:
            df = self.data['drugs']
            drug_lower = drug_name.lower()
            
            # Search in both drug_name and brand_names columns
            match = df[
                df['drug_name'].str.lower().str.contains(drug_lower, na=False) |
                df['brand_names'].str.lower().str.contains(drug_lower, na=False)
            ]
            
            if not match.empty:
                row = match.iloc[0]
                return self.format_drug_response(row)
        
        # Fallback to Gemini if not found in CSV
        return self.generate_drug_info(drug_name)

    def get_symptom_info(self, symptom_name):
        """Get symptom information from CSV data"""
        if 'symptoms' in self.data:
            df = self.data['symptoms']
            symptom_lower = symptom_name.lower()
            match = df[df['name'].str.lower().str.contains(symptom_lower, na=False)]
            if not match.empty:
                row = match.iloc[0]
                return self.format_symptom_response(row)
        
        # Fallback to Gemini
        return self.generate_symptom_info(symptom_name)

    # -------------------------------
    # FORMAT RESPONSES FROM CSV DATA
    # -------------------------------
    def format_condition_response(self, row):
        response = f"**🩺 {row.get('name', 'Condition')}**\n\n"
        response += f"**Description:** {row.get('description', 'N/A')}\n\n"
        response += f"**Common Symptoms:** {row.get('symptoms', 'N/A')}\n\n"
        response += f"**Treatment Options:** {row.get('treatment', 'N/A')}\n\n"
        response += f"**Prevention:** {row.get('prevention', 'N/A')}\n\n"
        response += f"**When to See a Doctor:** {row.get('when_to_see_doctor', 'N/A')}\n\n"
        response += "---\n*Information from medical database*"
        return response

    def format_drug_response(self, row):
        response = f"**💊 {row.get('drug_name', 'Medication').upper()}**\n\n"
        response += f"**Drug Class:** {row.get('drug_class', 'N/A')}\n\n"
        response += f"**Primary Uses:** {row.get('uses', 'N/A')}\n\n"
        response += f"**Common Side Effects:** {row.get('side_effects', 'N/A')}\n\n"
        response += f"**Contraindications:** {row.get('contraindications', 'N/A')}\n\n"
        response += f"**Important Precautions:** {row.get('precautions', 'N/A')}\n\n"
        response += f"**Available Forms:** {row.get('dosage_forms', 'N/A')}\n\n"
        response += f"**Brand Names:** {row.get('brand_names', 'N/A')}\n\n"
        response += "---\n*Information from medical database*"
        return response

    def format_symptom_response(self, row):
        response = f"**🔍 {row.get('name', 'Symptom')}**\n\n"
        response += f"**Description:** {row.get('description', 'N/A')}\n\n"
        response += f"**Possible Conditions:** {row.get('possible_conditions', 'N/A')}\n\n"
        response += f"**Severity Level:** {row.get('severity', 'N/A')}\n\n"
        response += f"**When to Seek Help:** {row.get('when_to_see_doctor', 'N/A')}\n\n"
        response += "---\n*Information from medical database*"
        return response

    # -------------------------------
    # GENERATE FALLBACK RESPONSES USING GEMINI
    # -------------------------------
    def generate_condition_info(self, condition_name):
        try:
            prompt = f"""
Provide concise, accurate information about the medical condition: {condition_name}

Structure with:
- Brief description
- Common symptoms
- General treatment approaches
- When to see a doctor

Include safety disclaimers and emphasize consulting healthcare professionals.
"""
            response = GEN_MODEL.generate_content(prompt)
            return response.text.strip() if response and response.text else f"Could not find information about {condition_name}."
        except Exception as e:
            return f"Error retrieving condition information: {e}"

    def generate_drug_info(self, drug_name):
        try:
            prompt = f"""
Provide concise, accurate information about the medication: {drug_name}

Structure with:
- Common uses
- Typical dosage guidelines
- Side effects
- Precautions

Include strong warnings to consult doctors before taking any medication.
"""
            response = GEN_MODEL.generate_content(prompt)
            return response.text.strip() if response and response.text else f"Could not find information about {drug_name}."
        except Exception as e:
            return f"Error retrieving drug information: {e}"

    def generate_symptom_info(self, symptom_name):
        try:
            prompt = f"""
Provide concise, accurate information about the symptom: {symptom_name}

Structure with:
- Description
- Possible causes
- When to seek medical attention
- General self-care tips

Include emergency warnings for serious symptoms.
"""
            response = GEN_MODEL.generate_content(prompt)
            return response.text.strip() if response and response.text else f"Could not find information about {symptom_name}."
        except Exception as e:
            return f"Error retrieving symptom information: {e}"

    # -------------------------------
    # DETECT QUERY TYPE
    # -------------------------------
    def detect_query_type(self, query):
        query_lower = query.lower()
        
        medication_keywords = ['medicine', 'medication', 'drug', 'pill', 'tablet', 'dose', 'dosage', 
                              'side effect', 'prescription', 'ibuprofen', 'aspirin', 'paracetamol']
        
        condition_keywords = ['symptom', 'condition', 'disease', 'illness', 'diagnosis', 'treatment',
                             'what is', 'have', 'suffering from', 'cancer', 'diabetes', 'asthma']
        
        symptom_keywords = ['pain', 'headache', 'fever', 'cough', 'nausea', 'vomiting', 'dizziness',
                           'rash', 'swelling', 'bleeding', 'shortness of breath']
        
        if any(keyword in query_lower for keyword in medication_keywords):
            return "medication"
        elif any(keyword in query_lower for keyword in condition_keywords):
            return "condition"
        elif any(keyword in query_lower for keyword in symptom_keywords):
            return "symptom"
        else:
            return "general"

    # -------------------------------
    # GENERATE SMART RESPONSE
    # -------------------------------
    def generate_smart_response(self, query):
        query_type = self.detect_query_type(query)
        
        if query_type == "medication":
            # Extract drug name
            drug_name = self.extract_entity(query, ['about', 'information on', 'tell me about'])
            return self.get_drug_info(drug_name or query)
        
        elif query_type == "condition":
            # Extract condition name
            condition_name = self.extract_entity(query, ['about', 'information on', 'tell me about'])
            return self.get_condition_info(condition_name or query)
        
        elif query_type == "symptom":
            # Extract symptom name
            symptom_name = self.extract_entity(query, ['about', 'information on', 'tell me about'])
            return self.get_symptom_info(symptom_name or query)
        
        else:
            # General query - try semantic search first
            results = self.semantic_search(query, top_k=3)
            if results:
                # Use the best matching result
                best_match = results[0]
                if best_match['type'] == 'conditions':
                    return self.format_condition_response(best_match['data'])
                elif best_match['type'] == 'drugs':
                    return self.format_drug_response(best_match['data'])
                elif best_match['type'] == 'symptoms':
                    return self.format_symptom_response(best_match['data'])
            
            # Fallback to Gemini for general questions
            try:
                prompt = f"""
Answer this medical question: "{query}"

Provide accurate, helpful information with appropriate safety disclaimers.
Always recommend consulting healthcare professionals for personal medical advice.
"""
                response = GEN_MODEL.generate_content(prompt)
                return response.text.strip() if response and response.text else "I couldn't generate a response for that question."
            except Exception as e:
                return f"Error generating response: {e}"

    def extract_entity(self, query, keywords):
        query_lower = query.lower()
        for keyword in keywords:
            if keyword in query_lower:
                return query_lower.split(keyword)[-1].strip()
        return query