"""
Task 2: FAQ Chatbot - WEB (browser) version   (topic: ShopEase online store)

Same NLP engine as the desktop version (NLTK preprocessing + TF-IDF + cosine
similarity) but the chat UI opens in your web browser.
Uses only Python's built-in web server - no Flask needed.

Install:  pip install nltk scikit-learn
Run:      python faq_chatbot_web.py      (browser opens automatically)
Stop:     press Ctrl+C in the terminal
"""

import json
import string
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(pkg, quiet=True)

# ======================= 1. FAQ DATA =======================
# (category, question, answer)
FAQS = [
    ("Orders", "How can I track my order?",
     "Go to 'My Orders' in your account and click 'Track'. You will also get a tracking link by email and SMS."),
    ("Orders", "How can I cancel my order?",
     "You can cancel from 'My Orders' before the order is shipped. After shipping, please use the return option."),
    ("Orders", "Can I change my delivery address?",
     "Yes, you can edit the address from 'My Orders' before the order is shipped."),
    ("Payments", "What payment methods do you accept?",
     "We accept credit/debit cards, JazzCash, Easypaisa, bank transfer, and Cash on Delivery."),
    ("Payments", "Do you have discount coupons or sales?",
     "Yes! Subscribe to our newsletter and follow our social media pages for coupon codes and seasonal sales."),
    ("Delivery", "How long does delivery take?",
     "Delivery takes 3-5 working days in major cities and 5-8 working days in other areas."),
    ("Delivery", "Do you offer free shipping?",
     "Yes, orders above Rs. 3000 qualify for free shipping."),
    ("Delivery", "Do you ship internationally?",
     "Currently we only deliver within Pakistan. International shipping will be available soon."),
    ("Returns & Refunds", "What is your return policy?",
     "You can return items within 7 days of delivery if they are unused and in original packaging."),
    ("Returns & Refunds", "How do I get a refund?",
     "Refunds are issued to your original payment method within 5-7 working days after we receive the returned item."),
    ("Returns & Refunds", "My product arrived damaged. What now?",
     "We are sorry about that. Contact support within 48 hours with photos and we will replace or refund it."),
    ("Account", "How do I create an account?",
     "Click 'Sign Up' at the top of the page, enter your name, email and password, then verify your email."),
    ("Account", "I forgot my password. What should I do?",
     "Click 'Forgot Password' on the login page and follow the link sent to your email to reset it."),
    ("Support", "How can I contact customer support?",
     "Email support@shopease.com or call 0800-12345, Monday to Saturday, 9 AM to 6 PM."),
    ("Support", "Are the products original?",
     "Yes, all products are 100% authentic and sourced directly from brands or authorized dealers."),
]

CATEGORIES = ["Orders", "Payments", "Delivery", "Returns & Refunds", "Account", "Support"]
CATEGORY_ICONS = {"Orders": "📦", "Payments": "💳", "Delivery": "🚚",
                  "Returns & Refunds": "↩️", "Account": "👤", "Support": "🎧"}

categories = [c for c, _, _ in FAQS]
questions = [q for _, q, _ in FAQS]
answers = [a for _, _, a in FAQS]

# ======================= 2. NLP PREPROCESSING =======================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english")) - {"not", "no"}


def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in string.punctuation and t not in stop_words and t.isalnum()]
    return " ".join(tokens)


# ======================= 3. MATCHING (TF-IDF + COSINE) =======================
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
faq_vectors = vectorizer.fit_transform([preprocess(q) for q in questions])
THRESHOLD = 0.25


def find_matches(user_input, top_n=3):
    """Return [(faq_index, similarity), ...] sorted best first."""
    cleaned = preprocess(user_input)
    if not cleaned:
        return []
    scores = cosine_similarity(vectorizer.transform([cleaned]), faq_vectors)[0]
    ranked = scores.argsort()[::-1][:top_n]
    return [(int(i), float(scores[i])) for i in ranked if scores[i] > 0]


SMALL_TALK = {
    "hi": "Hello! 👋 How can I help you today?",
    "hello": "Hello! 👋 How can I help you today?",
    "hey": "Hey there! 👋 What would you like to know?",
    "thanks": "You're welcome! 😊 Anything else I can help with?",
    "thank you": "You're welcome! 😊 Anything else I can help with?",
    "bye": "Goodbye! Happy shopping! 🛍️",
}


# ======================= 4. WEB PAGE (HTML + CSS + JS) =======================
PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ShopEase FAQ Assistant</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;font-family:'Segoe UI',Arial,sans-serif}
body{background:linear-gradient(135deg,#e0ecf8,#f4f6f8);min-height:100vh;display:flex;justify-content:center;align-items:center;padding:16px}
.app{width:100%;max-width:1000px;height:90vh;max-height:720px;background:#fff;border-radius:18px;box-shadow:0 12px 40px rgba(31,78,121,.25);display:flex;flex-direction:column;overflow:hidden}
header{background:linear-gradient(90deg,#1f4e79,#2e75b6);color:#fff;padding:14px 20px;display:flex;align-items:center;gap:14px}
header .logo{width:46px;height:46px;border-radius:50%;background:#fff;display:flex;align-items:center;justify-content:center;font-size:26px}
header h1{font-size:19px} header p{font-size:12px;color:#cfe2f3}
.online{margin-left:auto;font-size:12px;display:flex;align-items:center;gap:6px}
.online i{width:9px;height:9px;border-radius:50%;background:#4ade80;display:inline-block}
.body{flex:1;display:flex;min-height:0}
aside{width:210px;background:#f7f9fc;border-right:1px solid #e3e9f0;padding:14px 10px;overflow-y:auto}
aside h3{font-size:12px;color:#5b6b7c;text-transform:uppercase;letter-spacing:.6px;margin:0 6px 10px}
.topic{display:flex;align-items:center;gap:10px;width:100%;background:#fff;border:1px solid #e3e9f0;border-radius:10px;padding:10px;margin-bottom:8px;cursor:pointer;font-size:14px;color:#1f3a56;transition:.15s}
.topic:hover{background:#dbeafe;transform:translateX(3px)}
.topic span{font-size:20px}
.clear{background:#fdecea;border-color:#f5c6c0;color:#a33}
main{flex:1;display:flex;flex-direction:column;min-width:0}
#chat{flex:1;overflow-y:auto;padding:18px;display:flex;flex-direction:column;gap:12px;background:#fbfcfe}
.row{display:flex;gap:10px;align-items:flex-end;max-width:85%;animation:pop .25s ease}
.row.user{align-self:flex-end;flex-direction:row-reverse}
.avatar{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:18px;flex:none}
.bot .avatar{background:#dbeafe}.user .avatar{background:#1f4e79;color:#fff}
.bubble{padding:10px 14px;border-radius:16px;line-height:1.45;font-size:14.5px}
.bot .bubble{background:#e8eef6;color:#1b2a3a;border-bottom-left-radius:4px}
.user .bubble{background:#1f4e79;color:#fff;border-bottom-right-radius:4px}
.meta{font-size:11px;color:#7a8a99;margin-top:5px}
.typing span{display:inline-block;width:7px;height:7px;margin:0 2px;background:#7a8a99;border-radius:50%;animation:b 1s infinite}
.typing span:nth-child(2){animation-delay:.15s}.typing span:nth-child(3){animation-delay:.3s}
@keyframes b{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-6px)}}
@keyframes pop{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.chips{padding:8px 18px 0;background:#fbfcfe}
.chips label{font-size:12px;font-weight:600;color:#4a5a6a;display:block;margin-bottom:6px}
.chip{display:inline-block;border:1px solid #bfdbfe;background:#dbeafe;color:#1e3a8a;border-radius:18px;padding:7px 13px;margin:0 6px 6px 0;font-size:13px;cursor:pointer;transition:.15s}
.chip:hover{background:#1f4e79;color:#fff}
form{display:flex;gap:10px;padding:12px 18px 16px;background:#fbfcfe}
input{flex:1;border:1px solid #c9d6e4;border-radius:24px;padding:12px 18px;font-size:15px;outline:none}
input:focus{border-color:#2e75b6;box-shadow:0 0 0 3px rgba(46,117,182,.15)}
button.send{border:0;background:#1f4e79;color:#fff;border-radius:24px;padding:0 24px;font-size:15px;font-weight:600;cursor:pointer}
button.send:hover{background:#163a5a}
@media(max-width:700px){aside{display:none}}
</style></head><body>
<div class="app">
 <header><div class="logo">&#128717;&#65039;</div>
  <div><h1>ShopEase FAQ Assistant</h1><p>Ask about orders, payments, delivery, returns, account or support</p></div>
  <div class="online"><i></i> Online</div></header>
 <div class="body">
  <aside><h3>Browse by topic</h3><div id="topics"></div>
   <button class="topic clear" onclick="resetChat()"><span>&#128465;&#65039;</span>Clear chat</button></aside>
  <main>
   <div id="chat"></div>
   <div class="chips"><label id="chipLabel"></label><div id="chips"></div></div>
   <form onsubmit="send(event)"><input id="msg" placeholder="Type your question here..." autocomplete="off" autofocus>
    <button class="send" type="submit">Send &#10148;</button></form>
  </main>
 </div>
</div>
<script>
const chat=document.getElementById('chat'),chipsEl=document.getElementById('chips'),chipLabel=document.getElementById('chipLabel');
let busy=false;
function addMsg(who,text,meta){
  const r=document.createElement('div');r.className='row '+who;
  r.innerHTML='<div class="avatar">'+(who==='bot'?'\\u{1F916}':'\\u{1F642}')+'</div><div><div class="bubble"></div>'+(meta?'<div class="meta"></div>':'')+'</div>';
  r.querySelector('.bubble').textContent=text; if(meta)r.querySelector('.meta').textContent=meta;
  chat.appendChild(r);chat.scrollTop=chat.scrollHeight;return r;}
function setChips(label,items){chipLabel.textContent=label;chipsEl.innerHTML='';
  items.forEach(it=>{const c=document.createElement('span');c.className='chip';c.textContent=it.q;c.onclick=()=>ask({index:it.i},it.q);chipsEl.appendChild(c);});}
async function ask(payload,shown){
  if(busy)return;busy=true;
  addMsg('user',shown||payload.text);
  const t=addMsg('bot','');t.querySelector('.bubble').innerHTML='<span class="typing"><span></span><span></span><span></span></span>';
  const res=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const d=await res.json();await new Promise(r=>setTimeout(r,450));
  t.remove();addMsg('bot',d.answer,d.meta);if(d.chips)setChips(d.label,d.chips);busy=false;}
function send(e){e.preventDefault();const i=document.getElementById('msg');const v=i.value.trim();if(!v)return;i.value='';ask({text:v});}
function resetChat(){chat.innerHTML='';start();}
async function start(){
  const d=await (await fetch('/init')).json();
  const tp=document.getElementById('topics');
  if(!tp.children.length)d.categories.forEach(c=>{const b=document.createElement('button');b.className='topic';b.innerHTML='<span>'+c.icon+'</span>'+c.name;
    b.onclick=()=>ask({category:c.name},c.icon+' '+c.name);tp.appendChild(b);});
  addMsg('bot',d.welcome);setChips('Popular questions:',d.popular);}
start();
</script></body></html>"""

# ======================= 5. SERVER =======================
def chips_for(indices):
    return [{"i": i, "q": questions[i]} for i in indices]


def related(idx):
    return [i for i, c in enumerate(categories) if c == categories[idx] and i != idx][:3]


def handle_ask(data):
    if "category" in data:
        cat = data["category"]
        idxs = [i for i, c in enumerate(categories) if c == cat]
        return {"answer": f"Here are common questions about {cat}. Click one below!",
                "chips": chips_for(idxs), "label": f"{cat} questions:"}
    if "index" in data:
        i = int(data["index"])
        return {"answer": answers[i], "meta": f"Topic: {categories[i]}  |  Match: 100%",
                "chips": chips_for(related(i)), "label": "Related questions:"}
    text = data.get("text", "").strip()
    small = SMALL_TALK.get(text.lower().strip("!. "))
    if small:
        return {"answer": small}
    matches = find_matches(text)
    if matches and matches[0][1] >= THRESHOLD:
        i, score = matches[0]
        return {"answer": answers[i], "meta": f"Topic: {categories[i]}  |  Match: {score:.0%}",
                "chips": chips_for(related(i)), "label": "Related questions:"}
    guesses = [i for i, s in matches if s > 0.05] or [0, 3, 5]
    return {"answer": "Sorry, I couldn't find an exact answer. Did you mean one of these? "
                      "(Or contact support@shopease.com)",
            "chips": chips_for(guesses), "label": "Did you mean:"}


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/init":
            data = {"welcome": "Welcome to ShopEase! I can answer questions about orders, payments, "
                               "delivery, returns & refunds, your account and support. Pick a topic on the "
                               "left, click a suggestion below, or type your own question.",
                    "categories": [{"name": c, "icon": CATEGORY_ICONS[c]} for c in CATEGORIES],
                    "popular": chips_for([0, 3, 5, 8])}
            self._send(json.dumps(data).encode("utf-8"), "application/json")
        else:
            self._send(PAGE.encode("utf-8"), "text/html")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length) or b"{}")
        self._send(json.dumps(handle_ask(data)).encode("utf-8"), "application/json")

    def log_message(self, *args):  # keep terminal clean
        pass


if __name__ == "__main__":
    PORT = 5000
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"ShopEase FAQ Chatbot running at {url}  (Ctrl+C to stop)")
    threading.Timer(1, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")