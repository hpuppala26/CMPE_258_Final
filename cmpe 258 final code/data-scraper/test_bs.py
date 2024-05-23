import time
from collections import deque
from multiprocessing import Pool
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from requests.sessions import Session

session = Session()


# time day, month, year, hours, minutes, seconds in human readable format, suitable to append to file name
def time_now():
    t = time.localtime()
    return f"{t.tm_mday}-{t.tm_mon}-{t.tm_year}_{t.tm_hour}-{t.tm_min}-{t.tm_sec}"


def is_in_form(soup):
    """Check if the current URL is within a form."""
    return bool(soup.find("form"))


def scrape_website(url_depth):
    url, depth = url_depth
    try:
        response = session.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a", href=True)
        absolute_links = [urljoin(url, link["href"]) for link in links]
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []
    return [url_depth] + [(link, depth + 1) for link in absolute_links]


def scrape_website_recursive(url, max_depth=4):
    visited = set()
    queue = deque([(url, 0)])
    links_to_write = [url]
    depths_searched = [0]

    with Pool(processes=4) as pool:
        while queue:
            print(f"\rLength of queue: {len(queue)}", end="")
            current_url, depth = queue.popleft()
            if depth > max_depth:
                continue
            if "http" not in current_url or "https" not in current_url:
                continue
            # print(f"\rLinks visited: {len(visited)}", end="")

            if current_url in visited:
                continue

            visited.add(current_url)
            response = session.get(current_url)
            soup = BeautifulSoup(response.text, "html.parser")
            if is_in_form(soup):
                print(f"Form found at {current_url} -- skipping")
                continue
            new_links = pool.map(scrape_website, [(current_url, depth)])[0]

            if not new_links:  # if [] continues
                continue
            links_to_write.append(new_links[0][0])
            depths_searched.append(new_links[0][1])
            queue.extend([link for link in new_links[1:] if link[0] not in visited])

    with open(f"links_scrapped_{time_now()}.txt", "w") as f:
        f.write("\n".join(links_to_write))


if __name__ == "__main__":
    url = "https://www.who.int"
    links = scrape_website_recursive(url, max_depth=2)
