import os
import zipfile
import json
from Tokenizer import Tokenizer
from bs4 import BeautifulSoup
import pickle
import math
from bs4 import XMLParsedAsHTMLWarning
import warnings
import hashlib
import requests
from merge import merge_index
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

FLUSH_SIZE = 18000

def flush_partial_index(inverted_index, partial_index_num):
    sorted_index = dict(sorted(inverted_index.items()))

    serializable_index = {
        token: [{"doc_id": did, "freq": freq} for (did, freq) in postings]
        for token, postings in sorted_index.items()
    }
    with open(f"index/partial_index_{partial_index_num}.pkl", "wb") as f:
        pickle.dump(serializable_index, f)

def build_partial_index():
    inverted_index = {}
    doc_list = []
    seen_hashes = set()
    link_graph = {} 
    url_to_id = {}
    global doc_id 
    doc_id = 0
    global partial_index_num
    partial_index_num = 0
    zip_path = "developer.zip"
    archive = zipfile.ZipFile(zip_path, 'r')

    for file_name in archive.namelist():
        if file_name.startswith("DEV/") and file_name.endswith(".json"):
            file = archive.open(file_name)
            try:
                raw_data = file.read()
                try:
                    meta = json.loads(raw_data.decode("utf-8"))
                    encoding = meta.get("encoding", "utf-8")
                    data = json.loads(raw_data.decode(encoding))
                except Exception as e:
                    print(f"Encoding error in {file_name}: {e}")
                    file.close()
                    continue

                content = data.get("content")
                url = data.get("url")
                if url and isinstance(url, str):
                    url = url.split("#")[0]

                if not url or not isinstance(url, str):
                    continue  
                if not content or not isinstance(content, str):
                    continue
                if len(content.strip()) < 50:
                    continue

                if content.count("<") > 2 and content.count(">") > 2:
                    soup = BeautifulSoup(content, "html.parser")
                    visible_text = soup.get_text()
                else:
                    visible_text = content

                doc_hash = hashlib.md5(visible_text.encode("utf-8")).hexdigest()
                if doc_hash in seen_hashes:
                    continue
                seen_hashes.add(doc_hash)

                tokenizer = Tokenizer()
                boosted_tokens = []

                if content.count("<") > 2 and content.count(">") > 2:
                    title_text = soup.title.get_text() if soup.title else ""
                    headers = " ".join(h.get_text() for h in soup.find_all(["h1", "h2", "h3"]))
                    bold = " ".join(b.get_text() for b in soup.find_all("b"))
                    boosted_text = " ".join([title_text, headers, bold])
                    boosted_tokens = tokenizer.tokenize(boosted_text)

                tokens = tokenizer.tokenize(visible_text)
                freq_map = tokenizer.computeWordFrequencies(tokens)
                bigrams = tokenizer.compute_ngrams(tokens, 2)
                for bigram in bigrams:
                    if bigram in freq_map:
                        freq_map[bigram] += 1
                    else:
                        freq_map[bigram] = 1

                for token in boosted_tokens:
                    if token in freq_map:
                        freq_map[token] += 2 

                try:
                    response = requests.head(url, allow_redirects=True, timeout=2)
                    accessible = response.status_code == 200
                except:
                    accessible = False

                if not accessible:
                    continue 

                print(f"Doc ID {doc_id} - URL: {url}")
                print(f"Preview: {visible_text[:100].strip()}")
                print()

                doc_list.append(url)
                url_to_id[url] = doc_id
                outlinks = set()
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"]
                    if href.startswith("http"):
                        clean_href = href.split("#")[0]
                        if clean_href in url_to_id:
                            outlinks.add(url_to_id[clean_href]) 

                link_graph[doc_id] = outlinks

                for token, freq in freq_map.items():
                    if token not in inverted_index:
                        inverted_index[token] = []
                    inverted_index[token].append((doc_id, freq)) 
                doc_id += 1

                if len(doc_list) % FLUSH_SIZE == 0:
                    flush_partial_index(inverted_index, partial_index_num)
                    inverted_index.clear()
                    partial_index_num += 1

            except Exception as e:
                print(f"Error reading {file_name}: {e}")
            file.close()

    if inverted_index:
        flush_partial_index(inverted_index, partial_index_num)

    archive.close()

    with open("index/doc_id_map.pkl", "wb") as f:
        pickle.dump(doc_list, f)

    with open("index/link_graph.pkl", "wb") as f:
        pickle.dump(link_graph, f)

    return len(doc_list)