from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from groq import Groq
import os
import re

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Put it in your environment or a .env file.")
client = Groq(api_key=GROQ_API_KEY)

scraped_data = {}
chat_history = []

def is_valid_url(url):

    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def clean_text(text):

    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def scrape_website(url, max_length=10000):

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
 
        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()
 
        title = soup.title.string if soup.title else "No title"
 
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|main'))
        
        if main_content:
            text = main_content.get_text()
        else:
            text = soup.get_text()

        text = clean_text(text)

        if len(text) > max_length:
            text = text[:max_length] + "..."

        links = []
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href:
                absolute_url = urljoin(url, href)
                link_text = link.get_text().strip()
                if link_text:
                    links.append({'url': absolute_url, 'text': link_text})
 
        headings = []
        for i in range(1, 7):
            for heading in soup.find_all(f'h{i}'):
                headings.append({
                    'level': i,
                    'text': heading.get_text().strip()
                })
        
        return {
            'success': True,
            'url': url,
            'title': title,
            'content': text,
            'word_count': len(text.split()),
            'links': links[:20],  
            'headings': headings[:15]  
        }
    
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': f"Failed to fetch URL: {str(e)}"
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Error scraping website: {str(e)}"
        }

def search_and_scrape(topic, num_results=3):
    
    try:

        search_url = f"https://html.duckduckgo.com/html/?q={topic}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        results = []
        for result in soup.find_all('div', class_='result')[:num_results]:
            title_elem = result.find('a', class_='result__a')
            if title_elem:
                url = title_elem.get('href')
                title = title_elem.get_text()

                scraped = scrape_website(url)
                if scraped['success']:
                    results.append({
                        'title': title,
                        'url': url,
                        'content': scraped['content'][:2000]  
                    })
        
        return {
            'success': True,
            'results': results,
            'query': topic
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': f"Search failed: {str(e)}"
        }

def query_groq_with_scraped_data(question, data, model="mixtral-8x7b-32768"):
 
    try:
  
        if 'content' in data:
 
            context = f"""Website: {data['title']}
URL: {data['url']}

Content:
{data['content'][:10000]}"""
        else:
            
            context = f"Search Results for: {data.get('query', 'N/A')}\n\n"
            for i, result in enumerate(data.get('results', [])[:3], 1):
                context += f"\n--- Result {i}: {result['title']} ---\n"
                context += f"URL: {result['url']}\n"
                context += f"{result['content']}\n"
        
        messages = [
            {
                "role": "system",
                "content": f"""You are a helpful assistant that answers questions based on scraped web content.

{context}

Instructions:
- Answer questions based on the provided web content
- Be concise and accurate
- Cite the source when relevant
- If the answer is not in the provided content, say so"""
            },
            {
                "role": "user",
                "content": question
            }
        ]

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.3,
            max_tokens=1024
        )
        
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        return f"Error querying AI: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def scrape():
    global scraped_data, chat_history
    
    data = request.json
    url = data.get('url', '').strip()
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    if not is_valid_url(url):
        return jsonify({'error': 'Invalid URL format'}), 400

    result = scrape_website(url)
    
    if result['success']:
        scraped_data = result
        chat_history = []
        return jsonify(result)
    else:
        return jsonify(result), 400

@app.route('/search-scrape', methods=['POST'])
def search_scrape():
    global scraped_data, chat_history
    
    data = request.json
    topic = data.get('topic', '').strip()
    
    if not topic:
        return jsonify({'error': 'Search topic is required'}), 400

    result = search_and_scrape(topic)
    
    if result['success']:
        scraped_data = result
        chat_history = []
        return jsonify(result)
    else:
        return jsonify(result), 400

@app.route('/chat', methods=['POST'])
def chat():
    global scraped_data, chat_history
    
    if not scraped_data:
        return jsonify({'error': 'Please scrape a website first'}), 400
    
    data = request.json
    question = data.get('question', '')
    model = data.get('model', 'mixtral-8x7b-32768')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    response = query_groq_with_scraped_data(question, scraped_data, model)
    
    return jsonify({
        'question': question,
        'answer': response
    })

@app.route('/clear', methods=['POST'])
def clear():
    global scraped_data, chat_history
    scraped_data = {}
    chat_history = []
    return jsonify({'message': 'Data cleared successfully'})

@app.route('/models', methods=['GET'])
def get_models():
 
    models = [
        {'id': 'mixtral-8x7b-32768', 'name': 'Mixtral 8x7B (Fast & Balanced)'},
        {'id': 'llama-3.3-70b-versatile', 'name': 'Llama 3.3 70B (Most Capable)'},
        {'id': 'llama-3.1-8b-instant', 'name': 'Llama 3.1 8B (Fastest)'},
        {'id': 'gemma2-9b-it', 'name': 'Gemma 2 9B (Efficient)'}
    ]
    return jsonify(models)

if __name__ == '__main__':
    app.run(debug=True, port=5001)