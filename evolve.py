import os
import psycopg2
import requests
import xml.etree.ElementTree as ET

DB_URL = os.getenv("DB_URL")

def harvest_science():
    # Science & Tech Research တွေကို ဇွတ်ရှာမယ် (Quantum, AI, Advanced Tech)
    url = "http://export.arxiv.org/api/query?search_query=all:quantum+OR+all:technology+OR+all:science&max_results=10"
    response = requests.get(url)
    
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        papers = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
            papers.append((title.strip(), summary.strip()))
        
        # Neon ထဲ ဇွတ်သိမ်းမယ်
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        for p in papers:
            cur.execute("INSERT INTO research_data (title, detail, harvested_at) VALUES (%s, %s, NOW())", p)
        conn.commit()
        print(f"✅ Captured {len(papers)} Research Papers. Natural Order Growing.")
    else:
        print("❌ Science Harvest Failed. Retrying in next cycle...")

harvest_science()
