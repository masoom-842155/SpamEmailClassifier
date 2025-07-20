import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import streamlit as st
import time
import re
import string
import mysql.connector
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .prediction-box-spam {
        background-color: #ffcdd2;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
        font-weight: bold;
        color: #c62828;
    }
    .prediction-box-ham {
        background-color: #c8e6c9;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
        font-weight: bold;
        color: #2e7d32;
    }
    .example-box {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Database connection function
def connect_to_database():
    try:
        connection = mysql.connector.connect(
            host="spamemailclassifier0.streamlit.app",
            user="root",
            password="masoom842155",
            database="spam_classifier"
        )
        return connection
    except mysql.connector.Error as err:
        st.error(f"Database connection error: {err}")
        return None

# Function to save feedback to database
def save_feedback_to_db(feedback_type, email, feedback_text):
    connection = connect_to_database()
    if connection:
        try:
            cursor = connection.cursor()
            
            # Create table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    feedback_type VARCHAR(50),
                    email VARCHAR(100),
                    feedback_text TEXT,
                    submission_date DATETIME
                )
            """)
            
            # Insert feedback data
            query = """
                INSERT INTO user_feedback (feedback_type, email, feedback_text, submission_date)
                VALUES (%s, %s, %s, %s)
            """
            values = (feedback_type, email, feedback_text, datetime.now())
            cursor.execute(query, values)
            
            connection.commit()
            return True
        except mysql.connector.Error as err:
            st.error(f"Database error: {err}")
            return False
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    return False

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shield.png", width=200)
    st.title("Spam Email Classifier")
    st.markdown("---")
    
    # Feedback section
    st.subheader("Provide Feedback")
    feedback_type = st.radio("How was your experience?", ["😃 Great", "😐 Okay", "😞 Poor"])
    feedback_email = st.text_input("Your email (optional):", placeholder="example@email.com")
    feedback_text = st.text_area("Share your thoughts (optional):", height=100)
    
    if st.button("Submit Feedback"):
        # Simplified feedback processing with minimal UI updates
        with st.spinner("Processing feedback..."):
            # Single progress update instead of multiple small ones
            progress_bar = st.progress(0)
            progress_bar.progress(100)
            
            # Save feedback to database
            db_success = save_feedback_to_db(
                feedback_type, 
                feedback_email, 
                feedback_text
            )
            
            # Single UI update with success message
            if db_success:
                st.success("Thank you for your valuable feedback! It has been saved to our database.")
            else:
                st.success("Thank you for your valuable feedback!")
                st.warning("Note: Could not save to database, but your feedback is appreciated.")
    
    st.markdown("---")

# Text preprocessing function
@st.cache_data
def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Feature engineering function
@st.cache_data
def extract_features(text):
    """Extract additional features from text"""
    features = {}
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    features['digits_count'] = sum(c.isdigit() for c in text)
    features['special_chars'] = sum(not c.isalnum() and not c.isspace() for c in text)
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    return features

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv("spam.csv")
    data.drop_duplicates(inplace=True)
    data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])
    
    # Apply preprocessing to messages
    data['Processed_Message'] = data['Message'].apply(preprocess_text)
    
    return data

data = load_data()
mess = data['Processed_Message']
cat = data['Category']

# Split data with stratification to ensure balanced classes
mess_train, mess_test, cat_train, cat_test = train_test_split(
    mess, cat, test_size=0.2, random_state=42, stratify=cat
)

# Create a pipeline with TF-IDF and Naive Bayes
@st.cache_resource
def train_model():
    # Create a pipeline with TF-IDF and Naive Bayes
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2
        )),
        ('classifier', MultinomialNB())
    ])
    
    # Define hyperparameters for grid search
    parameters = {
        'tfidf__max_features': [3000, 5000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'classifier__alpha': [0.1, 0.5, 1.0]
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline, parameters, cv=5, n_jobs=-1, verbose=0, scoring='accuracy'
    )
    
    # Train the model
    grid_search.fit(mess_train, cat_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(mess_test)
    accuracy = accuracy_score(cat_test, y_pred)
    
    return best_model, accuracy, grid_search.best_params_

# Main content
st.markdown("<h1 class='main-header'>Spam Email Classifier</h1>", unsafe_allow_html=True)
st.markdown("Analyze messages to determine if they're spam or legitimate communications.")

# Show loading spinner while training the model
with st.spinner('Training advanced model for better accuracy...'):
    model, model_accuracy, best_params = train_model()

# Create tabs for different sections
tab1, tab2 = st.tabs(["Message Analysis", "Examples"])

with tab1:
    st.markdown("<h2 class='sub-header'>Analyze Your Message</h2>", unsafe_allow_html=True)
    
    # Input area with better styling
    input_mess = st.text_area("Enter the message to analyze:", height=150, 
                              placeholder="Type or paste your message here...")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        predict_button = st.button('Analyze Message', use_container_width=True)
    
    # Prediction function with probability
    def predict(message):
        # Preprocess the input message
        processed_message = preprocess_text(message)
        
        # Get prediction and probability
        proba = model.predict_proba([processed_message])[0]
        result = model.predict([processed_message])
        
        return result[0], max(proba)
    
    # Display prediction with animation
    if predict_button and input_mess:
        with st.spinner('Analyzing message...'):
            # Add a small delay for effect
            time.sleep(1)
            
            # Get prediction and confidence
            output, confidence = predict(input_mess)
            
            # Display result with animation
            st.markdown("<h3>Analysis Result:</h3>", unsafe_allow_html=True)
            
            if output == "Spam":
                st.markdown(f"<div class='prediction-box-spam'>SPAM DETECTED</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='prediction-box-ham'>NOT SPAM</div>", unsafe_allow_html=True)
            
            # Show confidence with progress bar
            st.markdown(f"<h4>Confidence: {confidence*100:.2f}%</h4>", unsafe_allow_html=True)
            st.progress(float(confidence))

with tab2:
    st.markdown("<h2 class='sub-header'>Example Messages</h2>", unsafe_allow_html=True)
    
    # Add sample spam/ham messages with better descriptions
    examples = {
        "Spam Example 1": "WINNER!! You've won a $1000 gift card! Click here to claim your prize now >>>",
        "Spam Example 2": "URGENT: Your account has been compromised. Reply with your password to verify identity.",
        "Not Spam Example 1": "Hey, are we meeting for lunch today? I was thinking about that new place downtown.",
        "Not Spam Example 2": "Your appointment is confirmed for tomorrow at 2:30 PM. Please arrive 15 minutes early."
    }
    
    st.markdown("Select an example to see how the system classifies different types of messages:")
    
    example_choice = st.selectbox("Choose an example:", list(examples.keys()))
    
    # Display the example in a nice box
    st.markdown("<div class='example-box'>", unsafe_allow_html=True)
    st.markdown(f"**{example_choice}:**")
    st.markdown(f"_{examples[example_choice]}_")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("Analyze Example"):
        with st.spinner('Analyzing example message...'):
            # Add a small delay for effect
            time.sleep(0.8)
            
            # Get prediction and confidence
            output, confidence = predict(examples[example_choice])
            
            # Display result with animation
            if output == "Spam":
                st.markdown(f"<div class='prediction-box-spam'>SPAM DETECTED</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='prediction-box-ham'>NOT SPAM</div>", unsafe_allow_html=True)
            
            # Show confidence with progress bar
            st.markdown(f"<h4>Confidence: {confidence*100:.2f}%</h4>", unsafe_allow_html=True)
            st.progress(float(confidence))
