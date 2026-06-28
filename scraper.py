"""
scraper.py — Download senior CVs (PDFs or images) from swgiitkgp.org/cvrepo.
Extracts text using pdfplumber (PDFs) or Gemini Vision API (images).
Generates embeddings using Gemini API.

Run this locally to build senior_cvs.json for the CV Optimizer.

Usage:
    python scraper.py --url https://swgiitkgp.org/cvrepo --categories DATA Software
    
Requirements:
    pip install requests beautifulsoup4 pdfplumber pillow google-generativeai
"""

import os
import sys
import json
import time
import argparse
import io
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_FILE = "senior_cvs.json"
TEMP_FILES_DIR = ".cv_temp"
MAX_CVS = 15  # Limit per category
REQUEST_TIMEOUT = 15
SUPPORTED_FORMATS = {'.pdf', '.jpg', '.jpeg', '.png', '.webp'}


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction: PDFs
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> Optional[str]:
    """Extract text from PDF bytes."""
    if not pdfplumber:
        print("  ⚠️  pdfplumber not installed. Install: pip install pdfplumber")
        return None
    
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text_parts = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n".join(text_parts) if text_parts else None
    except Exception as e:
        print(f"  ⚠️  PDF extraction failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction: Images (using Gemini Vision)
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_image(image_bytes: bytes, api_key: str, filename: str = "") -> Optional[str]:
    """Extract text from image using Gemini Vision API (OCR)."""
    if not genai:
        print("  ⚠️  google-generativeai not installed. Install: pip install google-generativeai")
        return None
    
    if not api_key:
        print("  ⚠️  No Gemini API key. Cannot extract text from image.")
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Convert bytes to PIL Image for validation
        try:
            img = Image.open(io.BytesIO(image_bytes))
            print(f"    Image size: {img.size}, format: {img.format}")
        except Exception as e:
            print(f"  ⚠️  Cannot open image: {e}")
            return None
        
        # Send to Gemini Vision
        print(f"    📸 Extracting text from image using Vision API...")
        response = model.generate_content([
            "Extract ALL text from this CV/resume image. Return the full text content exactly as it appears, preserving structure (line breaks, bullet points, etc.). Do not add commentary, just extract the text.",
            {
                "mime_type": f"image/{_get_image_mime_type(filename, image_bytes)}",
                "data": image_bytes
            }
        ])
        
        if response.text:
            return response.text
        else:
            print(f"  ⚠️  Vision API returned no text")
            return None
    
    except Exception as e:
        print(f"  ⚠️  Image OCR failed: {e}")
        return None


def _get_image_mime_type(filename: str, data: bytes) -> str:
    """Detect image MIME type from filename or magic bytes."""
    ext = Path(filename).suffix.lower()
    
    mime_map = {
        '.jpg': 'jpeg',
        '.jpeg': 'jpeg',
        '.png': 'png',
        '.webp': 'webp',
        '.gif': 'gif',
    }
    
    if ext in mime_map:
        return mime_map[ext]
    
    # Fallback: detect from magic bytes
    if data[:3] == b'\xff\xd8\xff':
        return 'jpeg'
    elif data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    elif data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'webp'
    
    return 'jpeg'  # default


# ─────────────────────────────────────────────────────────────────────────────
# Gemini embedding
# ─────────────────────────────────────────────────────────────────────────────

def get_gemini_embedding(text: str, api_key: str) -> Optional[List[float]]:
    """Generate embedding using Gemini API."""
    if not genai:
        print("  ⚠️  google-generativeai not installed.")
        return None
    
    if not text or len(text.strip()) < 50:
        print("  ⚠️  Text too short to embed.")
        return None
    
    try:
        genai.configure(api_key=api_key)
        response = genai.embed_content(
            model="models/embedding-001",
            content=text[:10000],  # Limit to first 10k chars
        )
        return response['embedding']
    except Exception as e:
        print(f"  ⚠️  Embedding failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Web scraping
# ─────────────────────────────────────────────────────────────────────────────

def scrape_media_links(base_url: str, categories: List[str]) -> Dict[str, List[Dict]]:
    """
    Scrape the website for CV links (PDFs or images) in given categories.
    
    Returns:
    {
        'DATA': [
            { 'name': 'cv1.pdf', 'url': 'https://...', 'company': 'Google', 'type': 'pdf' },
            { 'name': 'cv2.jpg', 'url': 'https://...', 'company': 'Amazon', 'type': 'image' },
        ],
        'Software': [...]
    }
    """
    print(f"\n🔍 Scraping {base_url}...")
    results = {cat: [] for cat in categories}
    
    try:
        print(f"  Fetching page...")
        response = requests.get(base_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Debug: Print page structure
        print(f"  Page length: {len(response.text)} chars")
        
        # Look for all links and images
        all_links = soup.find_all('a', href=True)
        all_imgs = soup.find_all('img', src=True)
        
        print(f"  Found {len(all_links)} links, {len(all_imgs)} images")
        
        # ──── Process links (PDFs, etc.) ────
        for link in all_links:
            href = link['href']
            text = link.get_text(strip=True).lower()
            
            file_ext = Path(href).suffix.lower()
            if file_ext not in SUPPORTED_FORMATS:
                continue
            
            # Match category
            matched_category = None
            for cat in categories:
                if cat.lower() in text or cat.lower() in href.lower():
                    matched_category = cat
                    break
            
            if matched_category:
                full_url = urljoin(base_url, href)
                filename = href.split('/')[-1] or f'cv{file_ext}'
                company = text.replace(file_ext.replace('.', ''), '').strip() or 'Unknown'
                
                file_type = 'image' if file_ext in {'.jpg', '.jpeg', '.png', '.webp'} else 'pdf'
                
                results[matched_category].append({
                    'name': filename,
                    'url': full_url,
                    'company': company,
                    'type': file_type,
                })
        
        # ──── Process images ────
        for img in all_imgs:
            src = img['src']
            alt = img.get('alt', '').lower()
            
            file_ext = Path(src).suffix.lower()
            if file_ext not in {'.jpg', '.jpeg', '.png', '.webp'}:
                continue
            
            matched_category = None
            for cat in categories:
                if cat.lower() in alt or cat.lower() in src.lower():
                    matched_category = cat
                    break
            
            if matched_category:
                full_url = urljoin(base_url, src)
                filename = src.split('/')[-1] or 'cv.jpg'
                company = alt.replace('cv', '').strip() or 'Unknown'
                
                results[matched_category].append({
                    'name': filename,
                    'url': full_url,
                    'company': company,
                    'type': 'image',
                })
        
        # Print results
        print(f"\n✅ Found media files:")
        for cat, items in results.items():
            if items:
                print(f"   {cat}: {len(items)} files")
                for item in items[:3]:
                    print(f"      - {item['company']}: {item['type']} - {item['url'][:60]}...")
        
        return results
    
    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        import traceback
        traceback.print_exc()
        return results


# ──────────────────────────────────────────────��──────────────────────────────
# Download & process
# ─────────────────────────────────────────────────────────────────────────────

def download_and_process_cvs(
    media_links: Dict[str, List[Dict]],
    api_key: str,
    max_per_category: int = MAX_CVS,
) -> Dict[str, List[Dict]]:
    """
    Download files (PDFs or images), extract text, generate embeddings.
    """
    
    os.makedirs(TEMP_FILES_DIR, exist_ok=True)
    
    processed = {}
    total_processed = 0
    
    for category, items in media_links.items():
        processed[category] = []
        
        for i, item in enumerate(items[:max_per_category]):
            print(f"\n📥 [{category}] {i+1}/{min(len(items), max_per_category)} ({item['type'].upper()})")
            print(f"   Company: {item['company']}")
            print(f"   URL: {item['url'][:80]}...")
            
            try:
                # Download
                response = requests.get(item['url'], timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                file_bytes = response.content
                
                print(f"   ✓ Downloaded ({len(file_bytes) / 1024:.1f} KB)")
                
                # Extract text based on type
                text = None
                if item['type'] == 'pdf':
                    text = extract_text_from_pdf(file_bytes)
                else:  # image
                    text = extract_text_from_image(file_bytes, api_key, item['name'])
                
                if not text:
                    print(f"   ⚠️  Could not extract text")
                    continue
                
                print(f"   ✓ Extracted text ({len(text)} chars)")
                
                # Generate embedding
                embedding = None
                if api_key:
                    embedding = get_gemini_embedding(text, api_key)
                    if embedding:
                        print(f"   ✓ Generated embedding ({len(embedding)} dims)")
                
                processed[category].append({
                    'company': item['company'],
                    'filename': item['name'],
                    'type': item['type'],
                    'url': item['url'],
                    'text': text,
                    'embedding': embedding,
                    'downloaded_at': datetime.now().isoformat(),
                })
                
                total_processed += 1
                
                # Rate limit
                time.sleep(1)
            
            except requests.exceptions.RequestException as e:
                print(f"   ❌ Download failed: {e}")
            except Exception as e:
                print(f"   ❌ Processing failed: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n\n✅ Processed {total_processed} CVs total")
    return processed


# ─────────────────────────────────────────────────────────────────────────────
# Save to JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_senior_cvs(data: Dict, output_file: str = OUTPUT_FILE):
    """Save processed CVs to JSON file."""
    
    total_cvs = sum(len(cvs) for cvs in data.values())
    total_embedded = sum(
        sum(1 for cv in cvs if cv.get('embedding')) 
        for cvs in data.values()
    )
    
    print(f"\n💾 Saving to {output_file}...")
    print(f"   Total CVs: {total_cvs}")
    print(f"   Embedded: {total_embedded}")
    
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_cvs': total_cvs,
                'total_embedded': total_embedded,
            },
            'cvs': data,
        }, f, indent=2)
    
    print(f"✅ Saved successfully!")
    print(f"📁 File size: {Path(output_file).stat().st_size / 1024:.1f} KB")


# ─────────────────────────────────────────────────────────────────────────────
# Main CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scrape senior CVs (PDFs or images) from swgiitkgp.org and embed them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape DATA and Software categories
  python scraper.py --url https://swgiitkgp.org/cvrepo --categories DATA Software
  
  # With custom API key
  python scraper.py --url https://swgiitkgp.org/cvrepo --categories DATA Software --api-key sk-...
  
  # Limit to 5 CVs per category
  python scraper.py --url https://swgiitkgp.org/cvrepo --categories DATA Software --max 5
        """
    )
    
    parser.add_argument(
        '--url',
        required=True,
        help='URL of the CV repository website'
    )
    parser.add_argument(
        '--categories',
        nargs='+',
        required=True,
        help='Categories to scrape (e.g., DATA Software)'
    )
    parser.add_argument(
        '--api-key',
        default=os.getenv('GEMINI_API_KEY'),
        help='Gemini API key (or set GEMINI_API_KEY env var)'
    )
    parser.add_argument(
        '--max',
        type=int,
        default=MAX_CVS,
        help=f'Max CVs to download per category (default: {MAX_CVS})'
    )
    parser.add_argument(
        '--output',
        default=OUTPUT_FILE,
        help=f'Output JSON file (default: {OUTPUT_FILE})'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎓 Senior CV Scraper & Embedder (PDFs + Images)")
    print("="*70)
    print(f"URL: {args.url}")
    print(f"Categories: {', '.join(args.categories)}")
    print(f"Max per category: {args.max}")
    print(f"Output: {args.output}")
    if args.api_key:
        print(f"✓ Gemini API key loaded")
    else:
        print(f"⚠️  No Gemini API key — embeddings will be skipped")
    print("="*70)
    
    # Step 1: Scrape
    media_links = scrape_media_links(args.url, args.categories)
    
    if not any(media_links.values()):
        print("\n❌ No media files found! Check:")
        print("   1. URL is correct")
        print("   2. Categories match the website structure")
        print("   3. Try inspecting the website in a browser first")
        return
    
    # Step 2: Download & process
    processed = download_and_process_cvs(media_links, args.api_key, args.max)
    
    # Step 3: Save
    save_senior_cvs(processed, args.output)
    
    print("\n" + "="*70)
    print("✅ Complete! Your CV Optimizer is ready to use senior CVs.")
    print("="*70)


if __name__ == '__main__':
    main()
