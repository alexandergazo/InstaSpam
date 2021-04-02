#!/usr/bin/env python3

import numpy as np
import io
import wikipedia as wiki
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import json
import urllib3
import praw


class RedditWrapper:
    def __init__(self, config):
        self.reddit = praw.Reddit(**config)

    def top_posts(self, subreddit, period, limit):
        iterator = self.reddit.subreddit(subreddit).top(period, limit=limit)
        return list(map(lambda x: (x.title, x.url), iterator))


class ImageDownloader:
    def __init__(self, unsplash_client_id):
        self.unsplash_id = unsplash_client_id
        self.https = urllib3.PoolManager()

    def download_wiki_images(self, query):
        # TODO try
        img_urls = wiki.page(wiki.search(query)[0]).images
        img_urls = filter(lambda x: x.split('.')[-1] != 'svg', img_urls)
        for url in img_urls:
            with open(url.split('/')[-1], "wb") as img_file:
                img_file.write(self.https.request('GET', url).data)

    def download_unsplash(self, query, limit):
        fields = {'client_id': self.unsplash_id,
                  'content_filter': 'high',
                  'query': query,
                  'per_page': limit}

        response = self.https.request('GET', 'https://api.unsplash.com/search/photos',
                                      fields=fields)

        img_urls = list(map(lambda x: x['urls']['full'].split('?')[0],
                            json.loads(response.data)['results']))

        dl_fields = {'w': 1500,
                     'h': 1500,
                     'fit': 'crop',
                     'ixlib': 'rb-1.2.1',
                     'q': 90,
                     'fm': 'png',
                     'cs': 'srgb'}

        crop_styles = ['entropy', 'edges']
        for idx, url in enumerate(img_urls):
            for crop_style in crop_styles:
                with open(f"img/{fields['query']}{idx}_{crop_style}.{dl_fields['fm']}",
                          "wb") as img:
                    response = self.https.request('GET', url,
                                                  fields=dl_fields | {'crop': crop_style})
                    img.write(response.data)


def create_annotation(reddit_title: str):
    words = reddit_title.split()
    assert words[0][:3] == 'TIL', 'Not TIL post.'
    words.pop(0)

    if words[0].lower() == 'that':
        words.pop(0)

    words[0] = words[0][0].upper() + words[0][1:]

    annotation = ' '.join(words)
    return annotation


def get_new_size(size, desired_size):
    ratio0 = desired_size[0] / size[0]
    ratio1 = desired_size[1] / size[1]
    ratio = max(ratio0, ratio1)
    return int(size[0] * ratio), int(size[1] * ratio)


def smart_resize(img, desired_size):
    smart_size = get_new_size(img.size, desired_size)
    return img.resize(smart_size)


def find_best_crop_entropy(img, crop_size):
    max_entropy = -np.inf
    max_x_offset = img.size[0] - crop_size[0]
    max_y_offset = img.size[1] - crop_size[1]
    for x_offset in range(0, max_x_offset + 1, (max_x_offset + 1) // 15 + 1):
        for y_offset in range(0, max_y_offset + 1, (max_y_offset + 1) // 15 + 1):
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


def main():
    with open("config.json", 'r') as config_file:
        config = json.load(config_file)

    reddit = RedditWrapper(config['reddit'])
    posts = reddit.top_posts('todayilearned', 'day', 2)

    img_dl = ImageDownloader(config['unsplash']['client_id'])

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

