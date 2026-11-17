"""Versioned, deterministic preparation shared by training and HTTP inference."""
import re
from html import unescape
from html.parser import HTMLParser

FIELDS = ('title', 'company_profile', 'description', 'requirements', 'benefits')
PREPROCESSING_VERSION = 'text-v1'
MAX_TEXT = 60000

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden += 1
        self.parts.append(' ')
    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.hidden = max(0, self.hidden - 1)
        self.parts.append(' ')
    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)

def clean_text(value):
    if value is None or not isinstance(value, str):
        return ''
    parser = TextExtractor()
    parser.feed(value)
    return re.sub(r'\s+', ' ', unescape(''.join(parser.parts))).strip()

def prepare_full_text(record):
    # No field names or labels are inserted: missing optional fields are empty.
    return ' '.join(filter(None, (clean_text(record.get(f)) for f in FIELDS)))

def prepare_text(record):
    # Historical text-v1 contract is retained for reproducible artifact loading.
    # HTTP/direct inference rejects over-limit full text rather than silently clipping.
    return prepare_full_text(record)[:MAX_TEXT]
