import time
from http.client import RemoteDisconnected

import numpy as np
import requests
import wikipedia as wiki
from PIL import Image
from PIL import ImageFilter
from tqdm import tqdm


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
            except (
                wiki.HTTPTimeoutError,
                requests.exceptions.ConnectionError,
                RemoteDisconnected,
            ):
                time.sleep(0.1)
        return (img_urls, page) if return_page else img_urls

    def download_wiki(self, query):
        img_urls = self._get_wiki_img_urls(query)
        img_urls = list(filter(lambda x: x.split('.')[-1] not in ['svg', 'gif'], img_urls))
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
        params = {
            'client_id': self.unsplash_id,
            'content_filter': 'high',
            'query': query,
            'per_page': limit,
        }

        response = requests.get('https://api.unsplash.com/search/photos', params=params)

        img_urls = list(map(lambda x: x['urls']['full'].split('?')[0], response.json()['results']))

        dl_params = {
            'w': 1500,
            'h': 1500,
            'fit': 'crop',
            'ixlib': 'rb-1.2.1',
            'q': 90,
            'fm': 'png',
            'cs': 'srgb',
        }

        crop_styles = ['entropy', 'edges', 'faces']
        for idx, url in enumerate(tqdm(img_urls)):
            for crop_style in crop_styles:
                response = requests.get(url, params=dl_params | {'crop': crop_style})
                with open(
                    f"img/{params['query']}{idx}_{crop_style}.{dl_params['fm']}", "wb"
                ) as img:
                    img.write(response.content)


def create_description(source_url, license):
    # s = 'Source: {}\n\nImage Credit: {}'
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
            mask[x_offset : x_offset + crop_size[0], y_offset : y_offset + crop_size[1]] = 255

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
        return (
            self._aux[box[2], box[3]]
            - self._aux[box[0], box[3]]
            - self._aux[box[2], box[1]]
            + self._aux[box[0], box[1]]
        )


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
