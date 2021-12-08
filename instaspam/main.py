#!/usr/bin/env python3
import json
import pickle
from collections import Counter

import numpy as np
import praw
import spacy
import tldextract
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from praw.models.reddit.more import MoreComments


# TODO maybe delete whole class
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
        iterator = self.reddit.subreddit('todayilearned').top(period, limit=n)

        results = []
        for sub in iterator:
            if sub.id in self.used:
                continue
            tup = (self._create_TIL_annotation(sub.title), sub.url, sub.id)
            results.append(tup)

        return results


def get_tf_comments(sub: list, nlp):
    counter = Counter()
    root_comments = filter(lambda x: not isinstance(x, MoreComments) and x.is_root, sub.comments)
    docs = map(lambda x: nlp(x.body), root_comments)
    for doc in docs:
        nouns = filter(
            lambda z: not z.is_stop and not z.is_punct and z.pos_ in ["NOUN", "PROPN"], doc
        )
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


def overlay_concept():
    img = Image.open('img/test.png')
    img2 = Image.new('RGBA', (1500, 1500), (255, 255, 0, 150))
    fnt = ImageFont.truetype('Arial', 40)
    d = ImageDraw.Draw(img2)
    d.text((700, 700), "TEST-TEST-TEST-TEST-TEST", font=fnt, fill=(0, 0, 0, 255))
    img.alpha_composite(img2)
    img.save('img/uuz.png')
    return img


def pseudo_method(USED_SET):
    FILENAME = "outs.json"
    with open(FILENAME) as f:
        j = json.load(f)
    not_reviewed, usable = [], []
    for post in j:
        if post['usable'] is None:
            not_reviewed.append(post)
        else:
            USED_SET.add(post['id'])
            if post['usable']:
                usable.append(post)
    with open(FILENAME, 'w') as f:
        json.dump(not_reviewed, f, indent=4)


def main():
    with open("config.json") as config_file:
        config = json.load(config_file)

    # TODO check if exists
    with open(config['reddit']['used_posts_set'], 'rb') as used_set_file:
        used_set_reddit = pickle.load(used_set_file)

    nlp = spacy.load("en_core_web_sm")

    reddit = praw.Reddit(**config['reddit']['login'])
    posts = list(reddit.subreddit('todayilearned').top('month', limit=40))
    posts = list(filter(lambda post: post.id not in used_set_reddit, posts))
    comment_keywords = list(
        map(lambda post: get_tf_comments(post, nlp).most_common(1)[0][0], posts)
    )

    post_titles = [create_TIL_annotation(post.title) for post in posts]
    title_docs = list(map(lambda title: nlp(title), post_titles))
    title_keywords = map(lambda doc: doc.ents[0].text if len(doc.ents) != 0 else '', title_docs)

    comb_keywords_ent, comb_keywords_nc = [], []
    for title_doc, comment_kw in zip(title_docs, comment_keywords):
        added = False
        for ent in title_doc.noun_chunks:
            if comment_kw in ent.text.lower():
                comb_keywords_nc.append(ent.text)
                added = True
                break
        if not added:
            comb_keywords_nc.append(comment_kw)
        added = False
        for ent in title_doc.ents:
            if comment_kw in ent.text.lower():
                comb_keywords_ent.append(ent.text)
                added = True
                break
        if not added:
            comb_keywords_ent.append(comment_kw)

    j = []
    for post, title, comm_kw, title_kw, comb_kw_ent, comb_kw_nc in zip(
        posts, post_titles, comment_keywords, title_keywords, comb_keywords_ent, comb_keywords_nc
    ):
        print(title)
        print('Title:  \t', title_kw)
        print('Comment:\t', comm_kw)
        print('ENT:    \t', comb_kw_ent)
        print('NC:     \t', comb_kw_nc)
        print()
        if tldextract.extract(post.url).domain == "wikipedia":
            wiki_kw = post.url.split('/')[-1].split('#')[0].replace('_', ' ').split('?')[0]
        keywords = [wiki_kw, title_kw, comm_kw, comb_kw_ent, comb_kw_nc]
        j.append(
            {
                'text': title,
                'suggested_keywords': np.unique(keywords).tolist(),
                'usable': None,
                'id': post.id,
                'url': post.url,
            }
        )
        # TODO json then another file which reads the json and creates keywords then another program which downloades the stuff. Idk how to rewrite the title if wrong. also add id to the json
    with open('outs.json', 'w') as f:
        json.dump(j, f, indent=4)

    # --------------- UNUSED TEST CODE -------------------
    # TO BE REMOVED / REIMPLEMENTED IN NEXT REFACTOR
    # keywords = []
    # for post in posts:
    #     if tldextract.extract(post[1]).domain == "wikipedia":
    #         keywords.append(post[1].split('/')[-1].split('#')[0].replace('_', ' ').split('?')[0])
    #         continue
    #     doc = nlp(post[0])
    #     print(post[0])
    #     print("Noun chunks", [(nc.text, nc.label_) for nc in doc.noun_chunks])
    #     useless = ['DATE', 'MONEY', 'TIME', 'CARDINAL', 'PERCENT', 'QUANTITY', 'NORP']
    #     ents = list(filter(lambda e: e.label_ not in useless, doc.ents))
    #     print("Entities: ", [(e.text, e.label_) for e in ents])
    #     from collections import Counter

    #     print(
    #         Counter(
    #             [
    #                 tok.lemma_
    #                 for tok in doc
    #                 if not tok.is_stop and not tok.is_punct and tok.pos_ == "NOUN"
    #             ]
    #         )
    #     )
    #     if ents == []:
    #         keyword = list(doc.noun_chunks)[0].text
    #     else:
    #         keyword = ents[0].text
    #     print(keyword)
    #     print()
    #     input()

    # img_dl = ImageDownloader(config['unsplash']['client_id'])
    # img_dl.download_wiki("church")
    # img = Image.open(f"img/{fields['query']}0_edges.{dl_fields['fm']}")
    # img = Image.open(io.BytesIO(response.data))

    # i = ImageDownloader(config['unsplash']['client_id'])
    # i.download_unsplash("friends", limit=3)


if __name__ == '__main__':
    main()
