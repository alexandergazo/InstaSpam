#!/usr/bin/env python3

from collections import Counter
import pickle
import tldextract
import spacy
from tqdm import tqdm
from http.client import RemoteDisconnected
import time
import numpy as np
import io
import wikipedia as wiki
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import json
import urllib
import requests
import praw
from praw.models.reddit.more import MoreComments


class LinkShortener:
    def __init__(self, config):
        self.api_key = config['api_key']
        self.api_url = 'http://cutt.ly/api/api.php?key={}&short={}&name{}'

    def shorten(self, url, name=''):
        url = urllib.parse.quote(url)
        r = requests.get(self.api_url.format(self.api_key, url, name))
        r = r.json()['url']
        if r['status'] == 7:
            return r
        return None


class RedditWrapper:
    def __init__(self, config: dict, used: set):
        self.reddit = praw.Reddit(**config)
        self.used = used

    # TODO maybe delete
    def top_posts(self, subreddit: str, period: str, limit: int):
        iterator = self.reddit.subreddit(subreddit).top(period, limit=limit)
        return list(map(lambda x: (x.title, x.url), iterator))

    def _create_TIL_annotation(self, title: str):
        words = title.split()
        assert words[0][:3] == 'TIL', 'Not TIL post.'
        words.pop(0)

        if words[0].lower() == 'that':
            words.pop(0)

        words[0] = words[0][0].upper() + words[0][1:]

        annotation = ' '.join(words)
        return annotation

    def TIL_top_processed(self, period: str, n: int):
        iterator = self.reddit.subreddit('todayilearned').top(perdiod, limit=n)

        results = []
        for sub in iterator:
            if sub.id in self.used:
                continue
            tup = (self._create_TIL_annotation(sub.title), sub.url, sub.id)
            results.append(tup)

        return results


class ImageDownloader:
    def __init__(self, unsplash_client_id):
        self.unsplash_id = unsplash_client_id

    def _get_wiki_img_urls(self, query, return_page=False):
        page_titles = None
        img_urls = None
        while img_urls is None:
            try:
                wiki.search.clear_cache()
                page_titles = page_titles or wiki.search(query)
                if page_titles == []:
                    # TODO suggest
                    return
                page_title = page_titles[0]
                page = wiki.page(page_title, auto_suggest=False)
                img_urls = page.images
            except wiki.exceptions.DisambiguationError as error:
                page = wiki.page(error.options[0], auto_suggest=False)
                img_urls = page.images
            except (wiki.HTTPTimeoutError, requests.exceptions.ConnectionError, RemoteDisconnected):
                    time.sleep(0.1)
        return (img_urls, page) if return_page else img_urls

    def download_wiki(self, query):
        img_urls = self._get_wiki_img_urls(query)
        img_urls = list(filter(lambda x: x.split('.')[-1] not in ['svg','gif'], img_urls))
        for url in tqdm(img_urls):
            response = None
            while response is None or response.status_code != 200:
                try:
                    response = requests.get(url)
                except (requests.exceptions.ConnectionError, RemoteDisconnected):
                    time.sleep(0.1)

            content_type, content_format = response.headers['Content-Type'].split('/', maxsplit=1)
            if content_type != 'image':
                continue

            with open(url.split('/')[-1], "wb") as img_file:
                img_file.write(response.content)

    def download_unsplash(self, query, limit=1):
        params = {'client_id': self.unsplash_id,
                  'content_filter': 'high',
                  'query': query,
                  'per_page': limit}

        response = requests.get('https://api.unsplash.com/search/photos', params=params)

        img_urls = list(map(lambda x: x['urls']['full'].split('?')[0],
                            response.json()['results']))

        dl_params = {'w': 1500,
                     'h': 1500,
                     'fit': 'crop',
                     'ixlib': 'rb-1.2.1',
                     'q': 90,
                     'fm': 'png',
                     'cs': 'srgb'}

        crop_styles = ['entropy', 'edges', 'faces']
        for idx, url in enumerate(tqdm(img_urls)):
            for crop_style in crop_styles:
                response = requests.get(url, params=dl_params | {'crop': crop_style})
                with open(f"img/{params['query']}{idx}_{crop_style}.{dl_params['fm']}",
                          "wb") as img:
                    img.write(response.content)


def create_description(source_url, license):
    s = 'Source: {}\n\nImage Credit: {}'
    return f'Source: {source_url}'


def get_new_size(size, desired_size):
    ratio0 = desired_size[0] / size[0]
    ratio1 = desired_size[1] / size[1]
    ratio = max(ratio0, ratio1)
    return int(size[0] * ratio), int(size[1] * ratio)


def smart_resize(img, desired_size):
    if isinstance(desired_size, int):
        desired_size = (desired_size, desired_size)
    smart_size = get_new_size(img.size, desired_size)
    return img.resize(smart_size)


def find_best_crop_entropy(img, crop_size, max_tries=(15, 15)):
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    if isinstance(max_tries, int):
        max_tries = (max_tries, max_tries)
    max_entropy = -np.inf
    max_x_offset = img.size[0] - crop_size[0]
    max_y_offset = img.size[1] - crop_size[1]
    for x_offset in range(0, max_x_offset + 1, (max_x_offset + 1) // max_tries[0] + 1):
        for y_offset in range(0, max_y_offset + 1, (max_y_offset + 1) // max_tries[1] + 1):
            mask = np.zeros(img.size)
            mask[x_offset:x_offset + crop_size[0],
                 y_offset:y_offset + crop_size[1]] = 255

            entropy = img.entropy(mask=Image.fromarray(mask.T.astype(np.uint8), 'L'))
            if entropy > max_entropy:
                max_entropy = entropy
                offset = (x_offset, y_offset)
    cropped = img.crop(offset + (offset[0] + crop_size[0], offset[1] + crop_size[1]))
    return cropped


class FastBoxSum:
    def __init__(self, array):
        self._aux = array.cumsum(axis=0).cumsum(axis=1)
        self._aux = np.pad(self._aux, [(1, 0), (1, 0)])

    def box_sum(self, box: tuple):
        return self._aux[box[2], box[3]] \
                   - self._aux[box[0], box[3]] \
                   - self._aux[box[2], box[1]] \
                   + self._aux[box[0], box[1]]


def find_best_crop_edge(img, crop_size):
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    max_edge_score = -np.inf
    img_data = np.asarray(img.convert('L').filter(ImageFilter.FIND_EDGES())).T
    fbs = FastBoxSum(img_data > 100)

    max_x_offset = img.size[0] - crop_size[0]
    max_y_offset = img.size[1] - crop_size[1]
    for x_offset in range(max_x_offset + 1):
        for y_offset in range(max_y_offset + 1):
            box = (x_offset, y_offset, x_offset + crop_size[0], y_offset + crop_size[1])
            edge_score = fbs.box_sum(box)

            if edge_score > max_edge_score:
                max_edge_score = edge_score
                offset = (x_offset, y_offset)
    cropped = img.crop(offset + (offset[0] + crop_size[0], offset[1] + crop_size[1]))
    return cropped


def get_tf_comments(sub: list, nlp):
    counter = Counter()
    root_comments = filter(lambda x: not isinstance(x, MoreComments) and x.is_root,
                           sub.comments)
    docs = map(lambda x: nlp(x.body), root_comments)
    for doc in docs:
        nouns = filter(lambda z: not z.is_stop and not z.is_punct and z.pos_ in ["NOUN", "PROPN"], doc)
        lemmas = list(map(lambda x: x.lemma_.lower(), nouns))
        counter.update(lemmas)
    return counter


def create_TIL_annotation(title: str):
    words = title.split()
    assert words[0][:3] == 'TIL', 'Not TIL post.'
    words.pop(0)

    if words[0].lower() == 'that':
        words.pop(0)

    words[0] = words[0][0].upper() + words[0][1:]

    annotation = ' '.join(words)
    return annotation


def main():
    with open("config.json", 'r') as config_file:
        config = json.load(config_file)

    # TODO check if exists
    with open(config['reddit']['used_posts_set'], 'rb') as used_set_file:
        used_set_reddit = pickle.load(used_set_file)

    nlp = spacy.load("en_core_web_sm")
    # i = ImageDownloader(config['unsplash']['client_id'])
    # i.download_unsplash("friends", limit=3)
    # return

    # TODO maybe delete whole class
    # reddit = RedditWrapper(config['reddit']['login'], used_set_reddit)
    reddit = praw.Reddit(**config['reddit']['login'])
    posts = list(reddit.subreddit('todayilearned').top('month', limit=20))
    posts = list(filter(lambda post: post.id not in used_set_reddit, posts))
    comment_keywords = list(map(lambda post: get_tf_comments(post, nlp).most_common(1)[0][0], posts))
    print(comment_keywords)

    post_titles = [create_TIL_annotation(post.title) for post in posts]
    title_docs = list(map(lambda title: nlp(title), post_titles))
    title_keywords = map(lambda doc: doc.ents[0].text if len(doc.ents) != 0 else '', title_docs)
    print(list(title_keywords))

    comb_keywords = []
    for title_doc, comment_kw in zip(title_docs, comment_keywords):
        added = False
        for ent in title_doc.ents:
            if comment_kw in ent.text.lower():
                comb_keywords.append(ent.text)
                added = True
                break
        if not added:
            comb_keywords.append(comment_kw)

    print(comb_keywords)
    return

    keywords = []
    for post in posts:
        if tldextract.extract(post[1]).domain == "wikipedia":
            keywords.append(post[1].split('/')[-1].split('#')[0].replace('_', ' ').split('?')[0])
            continue
        doc = ner(post[0])
        print(post[0])
        print("Noun chunks", [(nc.text, nc.label_) for nc in doc.noun_chunks])
        useless = ['DATE', 'MONEY', 'TIME', 'CARDINAL', 'PERCENT', 'QUANTITY', 'NORP']
        ents = list(filter(lambda e: e.label_ not in useless, doc.ents))
        print("Entities: ", [(e.text, e.label_) for e in ents])
        from collections import Counter
        print(Counter([tok.lemma_ for tok in doc if not tok.is_stop and not tok.is_punct and tok.pos_ == "NOUN"]))
        if ents == []:
            keyword = list(doc.noun_chunks)[0].text
        else:
            keyword = ents[0].text
        print(keyword)
        print()
        input()
    return

    img_dl = ImageDownloader(config['unsplash']['client_id'])
    img_dl.download_wiki("church")
    return

    ### DEBUG ###
    img = Image.open(f"img/{fields['query']}0_edges.{dl_fields['fm']}")
    img = Image.open(io.BytesIO(response.data))
    img2 = Image.open('img/test.png')
    img3 = Image.new('RGBA', (1500, 1500), (255, 255, 0, 150))
    fnt = ImageFont.truetype('Arial', 40)
    d = ImageDraw.Draw(img3)
    d.text((700,700),"CAAAAAAAAAAU", font=fnt, fill=(0, 0, 0, 255))
    # img.alpha_composite(img2)
    img.alpha_composite(img3)
    img.save('img/uuz.png')


if __name__ == '__main__':
    main()

