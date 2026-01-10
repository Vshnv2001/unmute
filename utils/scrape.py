import os
import time
import json
from urllib.parse import urlparse, quote, urljoin
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
from tqdm import tqdm
from requests.adapters import HTTPAdapter, Retry

# Selenium imports for dynamic content
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_URL   = "https://blogs.ntu.edu.sg/sgslsignbank/signs/"
OUTPUT_DIR = "sgsl_dataset"

# ——— 1) REQUESTS SESSION + RETRIES ———
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
session.mount("https://", HTTPAdapter(max_retries=retries))
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
})

# ——— 2) SELENIUM SETUP ———
chrome_opts = Options()
chrome_opts.add_argument("--headless")
chrome_opts.add_argument("--disable-gpu")
chrome_opts.add_argument("--no-sandbox")
driver = webdriver.Chrome(options=chrome_opts)
wait = WebDriverWait(driver, 10)

# ——— 3) URL BUILDER ———
def build_sign_url(raw_href):
    full = urljoin(BASE_URL, raw_href)
    parsed = urlparse(full)
    if parsed.query:
        key, val = parsed.query.split("=", 1)
        val_enc = quote(val, safe="")
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{key}={val_enc}"
    return full

# ——— 4) SCRAPING HELPERS ———
def get_sign_links():
    driver.get(BASE_URL)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "a.sign.btn.btn-red")))
    soup = BeautifulSoup(driver.page_source, "html.parser")
    raw_links = [a["href"] for a in soup.find_all("a", class_="sign btn btn-red")]
    return [build_sign_url(h) for h in raw_links]

def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name.strip())

# ——— 5) VARIANT SCRAPE ———
def scrape_variant(soup, variant_label=None):
    img_tag = soup.find("img", class_="w-100 img-fluid mb-2")
    gif_url = img_tag["src"].strip()
    if variant_label:
        name = sanitize_filename(variant_label)
    else:
        alt_text = img_tag.get("alt", "unknown").split("-")[0]
        name = sanitize_filename(alt_text)
    folder = os.path.join(OUTPUT_DIR, name)
    os.makedirs(folder, exist_ok=True)

    # download GIF
    gif_bytes = session.get(gif_url, timeout=10).content
    with open(os.path.join(folder, f"{name}.gif"), "wb") as f:
        f.write(gif_bytes)

    # metadata container
    meta = {
        "sign": name,
        "gif_url": gif_url,
        "description": None,
        "visual_guide": None,
        "translation_equivalents": None,
        "parameters": {},
        "units": []
    }

    # extract description, visual guide, translation
    for label in ["Description of Sign", "Visual Guide", "Translation Equivalents"]:
        header = soup.find("h2", class_="h5 fw-bold", string=label)
        if header:
            p = header.find_next_sibling("p")
            if p:
                key = label.lower().replace(" ", "_")
                meta[key] = p.get_text(strip=True)

    # extract parameters
    params_h2 = soup.find("h2", class_="h5 mb-4 fw-bold", string="Parameters of Sign")
    if params_h2:
        table = params_h2.find_next("table")
        if table:
            for row in table.find("tbody").find_all("tr"):
                key = row.find("th").get_text(strip=True)
                cells = row.find_all("td")
                if len(cells) == 2:
                    dom  = cells[0].get_text(strip=True)
                    nond = cells[1].get_text(strip=True)
                elif len(cells) == 1:
                    dom = nond = cells[0].get_text(strip=True)
                else:
                    continue
                meta["parameters"][key] = {"Dominant Hand": dom, "Non-Dominant Hand": nond}

    # extract units
    units_h2 = soup.find("h2", class_="h5 mb-4 fw-bold", string="Units of Sign")
    if units_h2:
        ul = units_h2.find_next("ul")
        if ul:
            urls_attr = ul.get("urls")
            urls = [u.strip() for u in urls_attr.split(",")] if urls_attr else [img["src"].strip() for img in ul.find_all("img")]
            units_dir = os.path.join(folder, "units")
            os.makedirs(units_dir, exist_ok=True)
            li_tags = ul.find_all("li", class_="list-inline-item")
            for idx, img_src in enumerate(urls, start=1):
                ext = os.path.splitext(urlparse(img_src).path)[1] or ".png"
                fname = f"{idx}{ext}"
                img_data = session.get(img_src, timeout=10).content
                with open(os.path.join(units_dir, fname), "wb") as f:
                    f.write(img_data)
                step_txt = None
                if li_tags and len(li_tags) >= idx:
                    p = li_tags[idx-1].find("p", class_="text-center")
                    step_txt = p.get_text(strip=True) if p else None
                meta["units"].append({"step": step_txt or f"Step {idx}", "filename": os.path.join("units", fname)})

    # save metadata
    with open(os.path.join(folder, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    time.sleep(0.5)

# ——— 6) DOWNLOAD SIGN (ALL VARIANTS) ———
def download_sign(sign_url):
    driver.get(sign_url)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.row[id]")))
    page_soup = BeautifulSoup(driver.page_source, "html.parser")

    # find variant values
    group = page_soup.find("div", class_="btn-group-vertical")
    if group:
        variants = [inp.get("value") for inp in group.find_all("input", class_="btn-check") if inp.get("value")]
    else:
        parsed = urlparse(sign_url)
        variants = [parsed.query.split("=", 1)[1]]

    # scrape each variant section directly from HTML
    for var in variants:
        section = page_soup.find("div", id=var)
        if section:
            variant_soup = BeautifulSoup(str(section), "html.parser")
        else:
            # fallback: use full page soup
            variant_soup = page_soup
        scrape_variant(variant_soup, variant_label=sanitize_filename(var))

# ——— 7) MAIN LOOP ———
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # links = get_sign_links()
    links = [
        "https://blogs.ntu.edu.sg/sgslsignbank/word?frm-word=Three%20%28Numeral%29",
        "https://blogs.ntu.edu.sg/sgslsignbank/word?frm-word=Thunder",
        "https://blogs.ntu.edu.sg/sgslsignbank/word?frm-word=Thursday",
        "https://blogs.ntu.edu.sg/sgslsignbank/word?frm-word=Ticket",
        "https://blogs.ntu.edu.sg/sgslsignbank/word?frm-word=Time",
        "https://blogs.ntu.edu.sg/sgslsignbank/word?frm-word=Tiong%20Bahru%20%28Place%29",
        "https://blogs.ntu.edu.sg/sgslsignbank/word?frm-word=Tired",
        "https://blogs.ntu.edu.sg/sgslsignbank/word?frm-word=To%20the%20power%20of%20X%20%28Mathematics%29",
        "https://blogs.ntu.edu.sg/sgslsignbank/word?frm-word=Toa%20Payoh%20%28Place%29",
        "https://blogs.ntu.edu.sg/sgslsignbank/word?frm-word=Today",
        "https://blogs.ntu.edu.sg/sgslsignbank/word?frm-word=Toilet",
        "https://blogs.ntu.edu.sg/sgslsignbank/word?frm-word=Tomorrow",
        "https://blogs.ntu.edu.sg/sgslsignbank/word?frm-word=Tonight",
    ]

    # links = ["https://blogs.ntu.edu.sg/sgslsignbank/word/?frm-word=Egg"]

    print(f"Found {len(links)} base words.")
    total_signs_downloaded = 0
    failed_downloads = defaultdict(list)

    for url in tqdm(links, desc="Downloading signs"):
        try:
            driver.get(url)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.row[id]")))
            page_soup = BeautifulSoup(driver.page_source, "html.parser")

            # find variant values
            group = page_soup.find("div", class_="btn-group-vertical")
            if group:
                variants = [inp.get("value") for inp in group.find_all("input", class_="btn-check") if inp.get("value")]
            else:
                parsed = urlparse(url)
                variants = [parsed.query.split("=", 1)[1]]

            for var in variants:
                try:
                    section = page_soup.find("div", id=var)
                    if section:
                        variant_soup = BeautifulSoup(str(section), "html.parser")
                    else:
                        variant_soup = page_soup
                    scrape_variant(variant_soup, variant_label=sanitize_filename(var))
                    total_signs_downloaded += 1
                except Exception as ve:
                    failed_downloads[url].append(var)
                    print(f"[✗] Variant failed: {url} - {var} → {type(ve).__name__}: {ve}")

        except Exception as e:
            failed_downloads[url].append("base")
            print(f"[✗] URL failed: {url} → {type(e).__name__}: {e}")

    print("\n✅ Download complete.")
    print(f"Total variants downloaded: {total_signs_downloaded}")
    print(f"Failed URLs: {len(failed_downloads)}")
    if failed_downloads:
        print("Failed details:")
        for u, vs in failed_downloads.items():
            print(f" - {u} → variants: {vs}")
    driver.quit()


if __name__ == "__main__":
    main()
