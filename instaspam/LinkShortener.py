import urllib

import requests


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
