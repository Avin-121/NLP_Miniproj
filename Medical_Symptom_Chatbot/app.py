import streamlit as st
from pipeline import MedicalChatbot
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Medical Information Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.disclaimer {
    background-color: #fff3cd;
    border-left: 5px solid #ffc107;
    padding: 10px;
    margin: 10px 0;
    border-radius: 5px;
}
.emergency-warning {
    background-color: #f8d7da;
    border-left: 5px solid #dc3545;
    padding: 10px;
    margin: 10px 0;
    border-radius: 5px;
    font-weight: bold;
}
.chat-bubble {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 15px;
    margin: 10px 0;
    border-left: 4px solid #1f77b4;
}
.user-bubble {
    background-color: #d1ecf1;
    padding: 15px;
    border-radius: 15px;
    margin: 10px 0;
    text-align: right;
    border-right: 4px solid #17a2b8;
}
.database-info {
    background-color: #e8f5e8;
    border-left: 4px solid #28a745;
    padding: 8px;
    margin: 5px 0;
    border-radius: 5px;
    font-size: 0.8em;
}
.feature-card {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    border: 1px solid #dee2e6;
}
.quick-button {
    width: 100%;
    margin: 2px 0;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<div class="main-header">🏥 Medical Information Assistant</div>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Important Disclaimer:</strong> This application provides general medical information for educational purposes only. 
    It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or 
    other qualified health provider with any questions you may have regarding a medical condition.
    </div>
    """, unsafe_allow_html=True)
    
    # Emergency warning
    st.markdown("""
    <div class="emergency-warning">
    🚨 For medical emergencies, call your local emergency number immediately! Do not rely on this application for emergency situations.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Loading medical database..."):
            st.session_state.chatbot = MedicalChatbot()
            st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.title("💊 Medical Assistant")
        
        st.markdown("### 🔍 Quick Search")
        
        # Quick access buttons for conditions
        st.markdown("**🤒 Common Conditions**")
        if 'conditions' in st.session_state.chatbot.data:
            # Check if the dataframe exists and has data
            df = st.session_state.chatbot.data['conditions']
            if not df.empty and 'name' in df.columns:
                conditions = df['name'].head(6).tolist()
                for condition in conditions:
                    if st.button(f"🩺 {condition}", key=f"cond_{condition}", use_container_width=True):
                        st.session_state.user_input = f"Tell me about {condition}"
        
        # Quick access buttons for medications
        st.markdown("**💊 Common Medications**")
        if 'drugs' in st.session_state.chatbot.data:
            df = st.session_state.chatbot.data['drugs']
            if not df.empty:
                # Use 'drug_name' column for the new structure
                if 'drug_name' in df.columns:
                    drugs = df['drug_name'].head(6).tolist()
                elif 'name' in df.columns:  # Fallback to old structure
                    drugs = df['name'].head(6).tolist()
                else:
                    drugs = []
                
                for drug in drugs:
                    if st.button(f"💊 {drug}", key=f"drug_{drug}", use_container_width=True):
                        st.session_state.user_input = f"Information about {drug} medication"
        
        # Quick access buttons for symptoms
        st.markdown("**🔍 Common Symptoms**")
        if 'symptoms' in st.session_state.chatbot.data:
            df = st.session_state.chatbot.data['symptoms']
            if not df.empty and 'name' in df.columns:
                symptoms = df['name'].head(6).tolist()
                for symptom in symptoms:
                    if st.button(f"🔍 {symptom}", key=f"symptom_{symptom}", use_container_width=True):
                        st.session_state.user_input = f"Tell me about {symptom} symptom"
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 💬 Chat with Medical Assistant")
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-bubble">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Check if response came from database or AI
                if "---" in message["content"] and "*Information from medical database*" in message["content"]:
                    source_indicator = "📚 From Medical Database"
                    bubble_class = "database-info"
                else:
                    source_indicator = "🤖 AI Generated Response"
                    bubble_class = "chat-bubble"
                
                st.markdown(f"""
                <div class="{bubble_class}">
                    {source_indicator}
                </div>
                <div class="chat-bubble">
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # User input
        user_input = st.text_input(
            "Ask about medications, conditions, symptoms, or general health:",
            value=st.session_state.get('user_input', ''),
            placeholder="e.g., Tell me about diabetes, side effects of ibuprofen, headache symptoms...",
            key="user_input_widget"
        )
        
        # Action buttons
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            if st.button("🚀 Smart Analysis", type="primary", use_container_width=True):
                if user_input.strip():
                    # Add user message to chat
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    
                    # Generate response
                    with st.spinner("🔍 Analyzing your query..."):
                        response = st.session_state.chatbot.generate_smart_response(user_input)
                    
                    # Add assistant response to chat
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Clear input
                    st.session_state.user_input = ""
                    st.rerun()
                else:
                    st.warning("Please enter a question")
        
        with col1_2:
            if st.button("🔄 New Conversation", use_container_width=True):
                st.session_state.user_input = ""
                st.rerun()
    
    with col2:
        st.markdown("### 📁 Data Categories")
        
        st.markdown("""
        <div class="feature-card">
        <strong>💊 Medications Database</strong><br>
        Drug information, side effects, dosages, interactions
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
        <strong>🩺 Conditions Database</strong><br>
        Diseases, symptoms, treatments, prevention
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
        <strong>🔍 Symptoms Database</strong><br>
        Symptom analysis, possible causes, severity
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
        <strong>💡 Solutions Database</strong><br>
        First aid, home remedies, self-care
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        st.markdown("""
        Try asking:
        - *"What is diabetes?"*
        - *"Side effects of aspirin"*
        - *"Headache symptoms and causes"*
        - *"Treatment for common cold"*
        - *"Can I take ibuprofen with blood pressure meds?"*
        """)

if __name__ == "__main__":
    main()