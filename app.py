import os
import re
from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from docx import Document
from openai import OpenAI
from werkzeug.utils import secure_filename
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather
import requests
from bs4 import BeautifulSoup
import threading
import time

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))



# Initialize Twilio client with credentials from environment variables

twilio_client = Client(os.getenv('TWILIO_ACCOUNT_SID'), os.getenv('TWILIO_AUTH_TOKEN'))

TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# Global cache for document content and conversation history
document_cache = {}
conversation_history = {}

# Calendly link
CALENDLY_LINK = "https://calendly.com/ali-shehroz-19991/30min"

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1).lower() in ALLOWED_EXTENSIONS

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_file):
    text = ''
    try:
        with open(pdf_file, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + '\n'
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_file}: {str(e)}")
    return text

# Function to extract text from DOCX files
def extract_text_from_docx(docx_file):
    text = ''
    try:
        doc = Document(docx_file)
        for para in doc.paragraphs:
            text += para.text + '\n'
    except Exception as e:
        print(f"Error extracting text from DOCX {docx_file}: {str(e)}")
    return text



def load_documents():
    global document_cache
    if not document_cache:  # Only load if cache is empty
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.endswith('.pdf') or filename.endswith('.docx'):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                text = ''
                if filename.endswith('.pdf'):
                    text = extract_text_from_pdf(file_path)
                elif filename.endswith('.docx'):
                    text = extract_text_from_docx(file_path)
                if text.strip():  # Only add non-empty documents
                    document_cache[filename] = text
    return document_cache

# Function to extract search terms from a user question
def extract_search_terms(question):
    question = question.lower()
    question = re.sub(r'[^\w\s]', '', question)  # Remove punctuation
    terms = question.split()
    return [term for term in terms if len(term) >= 3]

# Function to generate regex patterns from search terms
def generate_regex_patterns(search_terms):
    return [re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE) for term in search_terms]

# Function to search documents using regex patterns
def search_documents_with_regex(patterns, documents):
    results = {}
    for filename, content in documents.items():
        total_matches = 0
        matched_contexts = []
        for pattern in patterns:
            matches = list(pattern.finditer(content))
            total_matches += len(matches)
            for match in matches[:5]:
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 100)
                context = content[start:end]
                matched_contexts.append(context.strip())
        if total_matches > 0:
            results[filename] = {
                'match_count': total_matches,
                'contexts': matched_contexts[:10]
            }
    return dict(sorted(results.items(), key=lambda item: item[1]['match_count'], reverse=True))

def get_ai_response(user_input, call_sid=None):
    documents = load_documents()

    # Get relevant context from documents
    search_terms = extract_search_terms(user_input)
    patterns = generate_regex_patterns(search_terms)
    search_results = search_documents_with_regex(patterns, documents)

    combined_context = ""
    if search_results:
        combined_context = "\n".join([f"From {filename}: " + "\n".join(result['contexts'])
                                        for filename, result in search_results.items()])

    # Build conversation history for context
    conversation_context = ""
    if call_sid and conversation_history.get(call_sid):
        conversation_context = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}"
                                            for msg in conversation_history[call_sid][-3:]])  # Use last 3 exchanges

    # Create a more conversational prompt
    prompt = """You are Sarah, a friendly and helpful representative from MultipleAI Solutions. 

You need to make a human like conversation. When you call start with some ice breaker and discuss about general based on user response. Then slowly move to business pitch. 

If you feel the conversation is ending or if the customer asks for it directly, push for appointment and send the appointment link.

The main thing is to have a meaningful conversation

DO NOT REPEATT YOURSELF, Don't say Hi how you doing or such again and again

Examples:
- User: "I'm having trouble automating my customer service with AI."
  Bot: "I understand that can be challenging. We can definitely help with that. Would you like to schedule an appointment to discuss your specific needs? [Appointment Suggested]"
- User: "Tell me more about your AI solutions for data analysis."
  Bot: "We offer a range of data analysis solutions, from predictive modeling to real-time insights. Would you like to set up a meeting to explore how we can tailor these solutions to your business? [Appointment Suggested]"
- User: "I'm interested in learning how AI can improve my business efficiency."
  Bot: "That's a great question. We have several ways we can improve your bussiness efficency. Would you like to book a call, and we can discuss the best options for your business? [Appointment Suggested]"
- User: "Okay, that sounds good." (After a few exchanges about AI services)
  Bot: "Great! Let's schedule a time to chat. [Appointment Suggested]"

Previous conversation:
{conversation_context}

Relevant document information:
{combined_context}

User's question: {user_input}

Respond in a helpful, natural, and conversational way, and suggest an appointment when appropriate:"""

    try:
        # Use GPT-4 for better responses
        openai_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ],
            max_tokens=75,
            n=1,
            stop=None,
            temperature=0.7,
        )

        response_text = openai_response.choices[0].message.content.strip()


        # Check if the AI suggested an appointment
        suggested_appointment = False
        if "[Appointment Suggested]" in response_text:
            suggested_appointment = True
            response_text = response_text.replace("[Appointment Suggested]", "")

        # Save to conversation history if this is a call
        if call_sid:
            conversation_history[call_sid].append({
                "user": user_input,
                "assistant": response_text
            })

            # Limit conversation history size
            if len(conversation_history[call_sid]) > 10:
                conversation_history[call_sid] = conversation_history[call_sid][-10:]

        return {"response": response_text, "suggested_appointment": suggested_appointment}
    except Exception as e:
        print(f"Error in get_ai_response: {e}")
        return {"response": "I'm sorry, I'm having trouble processing your request right now. Could you try again?", "suggested_appointment": False}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files part in the request'})
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'})

    uploaded_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            uploaded_files.append(filename)

            # Load new document text and append to cache
            if filename.endswith('.pdf'):
                document_cache[filename] = extract_text_from_pdf(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            elif filename.endswith('.docx'):
                document_cache[filename] = extract_text_from_docx(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return jsonify({'files': uploaded_files})


from flask import jsonify
from flask import make_response
from markupsafe import Markup


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form.get('message', '').strip()
    if not user_message:
        return jsonify({'response': "Hi there! How can I help you today?"})

    ai_response = get_ai_response(user_message)
    response_text = ai_response["response"]
    suggested_appointment = ai_response["suggested_appointment"]

    if suggested_appointment:
        calendly_html = f'<a href="{CALENDLY_LINK}" target="_blank">schedule a meeting here</a>'
        response_text += f'\n\nTo schedule a meeting, please {calendly_html}'
        return jsonify({'response': Markup(response_text)})

    return jsonify({'response': response_text})

@app.route('/call', methods=['POST'])
def call_endpoint():
    phone_number = request.form.get('phone_number')
    if not phone_number:
        return jsonify({'error': 'No phone number provided'})
    return make_call(phone_number)

def make_call(phone_number):
    try:
        # Ensure the phone number is in E.164 format
        if not phone_number.startswith('+'):
            # If in US/Canada format, add +1
            if len(phone_number) == 10:
                phone_number = '+1' + phone_number
            else:
                # For other regions, add + but user should provide country code
                phone_number = '+' + phone_number

        # Full URL to your TwiML endpoint - update this with your actual ngrok URL
        twiml_url = "https://familiar-bernie-onewebonly-45eb6d74.koyeb.app/twiml"

        # Make the call
        call = twilio_client.calls.create(
            to=phone_number,
            from_=TWILIO_PHONE_NUMBER,
            url=twiml_url
        )

        # Initialize conversation history for this call
        conversation_history[call.sid] = []

        return jsonify({"success": True, "call_sid": call.sid})
    except Exception as e:
        print(f"Call Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/twiml', methods=['POST', 'GET'])
def twiml_response():
    """Initial TwiML response when call is first connected"""
    response = VoiceResponse()
    call_sid = request.values.get('CallSid')

    # Store the call SID in the session
    if call_sid:
        # Initialize conversation for this call
        conversation_history[call_sid] = []

    # Create a Gather verb with short timeout
    gather = Gather(
        input='speech dtmf',
        action='/conversation',
        method='POST',
        timeout=3,  # Short initial timeout
        speechTimeout='auto',
        bargeIn=True
    )

    # Short welcome message
    gather.say(
        'Hi there! This is Sarah from MultipleAI Solutions. How are you today?',
        voice='alice'
    )

    response.append(gather)

    # If no response, go to conversation endpoint anyway
    response.redirect('/conversation')

    return str(response)

@app.route('/conversation', methods=['POST'])
def conversation():
    """Main conversation handling endpoint"""
    # Get input from the user
    user_speech = request.values.get('SpeechResult', '')
    call_sid = request.values.get('CallSid')
    digits = request.values.get('Digits', '')

    # Create TwiML response
    response = VoiceResponse()

    # Handle hang up requests
    if digits == '9' or any(word in user_speech.lower() for word in ['goodbye', 'bye', 'hang up', 'end call']):
        response.say("Thank you for your time. Have a great day!", voice='alice')
        response.hangup()
        return str(response)

    # Default message if no input detected
    input_text = user_speech if user_speech else "Hello"
    if digits:
        input_text = f"Button {digits} pressed"

    # Get AI response based on input
    ai_response_data = get_ai_response(input_text, call_sid)
    ai_response = ai_response_data["response"] 

    # Check for booking keywords and send SMS
    if call_sid and any(keyword in input_text.lower() for keyword in ["book", "appointment", "schedule", "meeting"]):
        try:
            call = twilio_client.calls(call_sid).fetch()
            phone_number = call.to
            message = twilio_client.messages.create(
                body=f"Here is the link to schedule a meeting: {CALENDLY_LINK}",
                from_=TWILIO_PHONE_NUMBER,
                to=phone_number
            )
            print(f"SMS sent to {phone_number}: {message.sid}")
            ai_response += "\n\nI have also sent you an SMS with the booking link."
        except Exception as e:
            print(f"Error sending SMS: {e}")
            ai_response += "\n\nI encountered an error sending the booking link via SMS."

    # Create a Gather that allows for interruption
    gather = Gather(
        input='speech dtmf',
        action='/conversation',
        method='POST',
        timeout=5,
        speechTimeout='auto',
        bargeIn=True
    )

    # Say the AI response inside the Gather to allow interruption
    gather.say(ai_response, voice='alice')
    response.append(gather)
    response.pause(length=1) #small pause for user to interrupt
    final_gather = Gather(input = 'speech dtmf', action = '/conversation', method ='POST', timeout= 5, speechTimeout= 'auto', bargeIn =True)
    response.append(final_gather)

    #print conversation to terminal
    if call_sid:
        print(f"Call SID: {call_sid}")
        print(f"User: {input_text}")
        print(f"Sarah: {ai_response}")
        return str(response)

if __name__ == '__main__':
    app.run(debug=True)
