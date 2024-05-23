from multiprocessing import Pool

import requests
from tqdm import tqdm

links = [x.strip() for x in links]


def check_url(url):
    try:
        response = requests.head(url, allow_redirects=True)
        content_type = response.headers.get("content-type", "").lower()
        if "pdf" in content_type:
            return (url, 0)
        else:
            return (url, 1)
    except requests.exceptions.RequestException as e:
        return (url, "Error: {}".format(str(e)))


def process_urls(urls):
    with Pool() as pool:
        results = list(tqdm(pool.imap(check_url, urls), total=len(urls)))
    normal_links = [url for url, result in results if result == 1]
    pdf_links = [url for url, result in results if result == 0]
    return normal_links, pdf_links

if __name__ == "__main__":
    normal_links, pdf_links = process_urls(links)

    with open(
        "/Users/hrithikpuppala/Desktop/projects/cmpe-258/llm-interface/normal_links.txt", "w"
    ) as f:
        f.write("\n".join(normal_links))


    with open(
        "/Users/hrithikpuppala/Desktop/projects/cmpe-258/llm-interface/pdf_links.txt", "w"
    ) as f:
        f.write("\n".join(pdf_links))

    print("Done")
