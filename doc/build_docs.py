from bs4 import BeautifulSoup
import pypandoc
from collections import defaultdict
import os
from tqdm import tqdm

f = open('../docs/Num.html')
soup = BeautifulSoup(f, features='lxml')
f.close()


class Method:
    def __init__(self, html):
        self.html = html
        self.rst = None
        self.title = None

    def parse(self):
        sig = self.html.find('div', {'class': 'signature'})
        self.title = sig.find('strong').text
        link = sig.find('a', 'method-permalink')
        if link is not None:
            link.decompose()
        for a in sig.findAll('a'):
            del a['href']
        self.rst = pypandoc.convert_text(str(self.html), 'rst', format='html').replace('\r', '')

def find_methods(html):
    return [Method(d) for d in html.findAll('div', {'class': 'entry-detail'})]


methods = find_methods(soup)
for method in tqdm(methods):
    method.parse()


d = defaultdict(list)
for method in methods:
    d[method.title].append(method.rst)

for title, content in d.items():
    with open(f'reference/generated/{title}.rst', 'w', newline='\n', encoding='utf-8') as out:
        pad = '*' * len(title)
        out.write(f'{pad}\n')
        out.write(f'{title}\n')
        out.write(f'{pad}\n\n')
        out.write('\n\n'.join(content))
