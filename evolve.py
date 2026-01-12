import os
import psycopg2
import requests
import xml.etree.ElementTree as ET

# Database URL ကို Secrets ကနေ ဇွတ်ယူမယ်
DB_URL = os.getenv("DB_URL")

def harvest():
    # ArXiv ကနေ Science Data ဇွတ်ဆွဲမယ်
    url = "http://export.arxiv.org/api/query?search_query=all:ai&max_results=10"
    
    try:
        # Database ချိတ်မယ်
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        
        # Table မရှိရင် ဇွတ်ဆောက်မယ်
        cur.execute("CREATE TABLE IF NOT EXISTS research_data (id SERIAL PRIMARY KEY, title TEXT, harvested_at TIMESTAMP DEFAULT NOW());")

        # Data ဆွဲမယ်
        response = requests.get(url)
        root = ET.fromstring(response.content)
        entries = root.findall('{http://www.w3.org/2005/Atom}entry')
        
        print(f"Found {len(entries)} papers.")

        for entry in entries:
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
            # ဒေတာသွင်းမယ်
            cur.execute("INSERT INTO research_data (title) VALUES (%s)", (title,))
        
        # ဇွတ် SAVE လုပ်မယ် (ဒါမှ 0 ကနေ တက်မှာ)
        conn.commit()
        
        cur.close()
        conn.close()
        print("Success: Data saved to Neon.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    harvest()
