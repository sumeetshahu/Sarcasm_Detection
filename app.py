import re
import os
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Function to pre-process text
def preprocess(phrase): 
  
    phrase = phrase.lower()   
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub('[^\w\s]','', phrase).strip()

    return phrase

# loading tokenizer object
with open('tokenizer.pkl', 'rb') as f:
    t = pickle.load(f)
    
# loading best model
model = load_model('bi_lstm_model1.hdf5')

def predict(s):
    '''This function takes a comment(string) as input and 
       returns whether the comment is sarcastic or not as output'''
    
    # Convert input string to list
    inp_str = [preprocess(s)]

    # Tokenize input string
    encoded_str = t.texts_to_sequences(inp_str)

    # Padding input sequence to have length of 30
    padded_str = pad_sequences(encoded_str, maxlen=30, dtype='int32', 
                               padding='post', truncating='post', value=0.0)
    
    # prediction on padded input sequence
    pred = model.predict(padded_str).flatten()[0]
    pred_int = np.where(pred >= 0.5, 1, 0).flatten()[0]

    # Output string
    if pred_int == 1:
        prob = round(pred * 100, 2)
        op_str = 'The above comment is sarcastic with {} % confidence'.format(prob)  
    else:
        prob = round((1-pred) * 100, 2)
        op_str = 'The above comment is not sarcastic with {} % confidence'.format(prob)
    
    return op_str

def main():
    st.set_page_config(page_title="SARCASM DETECTION", 
                       page_icon=":robot_face:",
                       layout="wide",
                       )
    st.markdown("<h4 style='text-align: center; color:grey;'>Sarcasm Detection with NLP &#129302;</h4>", unsafe_allow_html=True)
    st.text('')
    st.markdown(f'<h3 style="text-align: left; color:#F63366; font-size:28px;">Detect Sarcasm</h3>', unsafe_allow_html=True)
    st.text('')
    input_text = st.text_area("Enter text and click on Predict to know if it's sarcastic", max_chars=500, height=150)
    if st.button("Predict"):
        output = predict(input_text)
        st.write(output)
 
if __name__=='__main__':
    main()

    