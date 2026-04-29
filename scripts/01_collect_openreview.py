import argparse
import os
from pathlib import Path
import sys
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import write_jsonl


def get_client():
    import openreview

    username = os.getenv('OPENREVIEW_USERNAME')
    password = os.getenv('OPENREVIEW_PASSWORD')
    if username and password:
        return openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net', username=username, password=password)
    return openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')


def label_from_decision(decision_text: str):
    if not decision_text:
        return None
    d = decision_text.lower()
    skip_terms = ['withdrawn', 'withdraw']
    accept_terms = ['accept', 'oral', 'spotlight', 'poster']
    reject_terms = ['reject', 'desk reject']
    if any(t in d for t in skip_terms):
        return None
    if any(t in d for t in reject_terms):
        return 0
    if any(t in d for t in accept_terms):
        return 1
    return None


def label_from_submission_content(content, venue_id: str):
    for key in ('decision', 'venue', 'venueid', 'venue_id'):
        value = extract_content_value(content, key, '')
        label = label_from_decision(str(value))
        if label is not None:
            return value, label

    venue_value = str(extract_content_value(content, 'venue', '')).lower()
    venueid_value = str(extract_content_value(content, 'venueid', '') or extract_content_value(content, 'venue_id', '')).lower()
    canonical = venue_id.lower()
    if venue_value or venueid_value:
        for text_value in (venue_value, venueid_value):
            if any(term in text_value for term in ('withdraw',)):
                return '', None
            if any(term in text_value for term in ('desk reject', 'reject', 'decline')):
                return extract_content_value(content, 'venue', '') or extract_content_value(content, 'venueid', ''), 0
        if canonical in venueid_value and 'submitted' not in venue_value:
            return extract_content_value(content, 'venue', '') or extract_content_value(content, 'venueid', ''), 1
    return '', None


def decision_from_replies(note):
    details = getattr(note, 'details', None) or {}
    replies = details.get('directReplies') or details.get('replies') or []
    for reply in replies:
        invitations = reply.get('invitations') or []
        invitation_text = ' '.join(invitations).lower()
        if 'decision' not in invitation_text:
            continue
        content = reply.get('content') or {}
        for key in ('decision', 'recommendation', 'venue'):
            value = extract_content_value(content, key, '')
            label = label_from_decision(str(value))
            if label is not None:
                return value, label
    return '', None


def extract_content_value(content, key, default=''):
    val = content.get(key, default)
    if isinstance(val, dict) and 'value' in val:
        return val['value']
    return val


def collect_venue(client, venue_id: str):
    invitations = [
        f'{venue_id}/-/Submission',
        f'{venue_id}/-/Blind_Submission',
        f'{venue_id}/-/Full_Submission',
    ]
    notes = []
    for invitation in invitations:
        try:
            notes = list(client.get_all_notes(invitation=invitation, details='directReplies'))
        except Exception as exc:
            print(f'WARN: failed invitation {invitation}: {exc}')
            continue
        if notes:
            print(f'using invitation {invitation} for {venue_id} ({len(notes)} notes)')
            break
    if not notes:
        raise RuntimeError(f'no submissions found for venue {venue_id} using invitations {invitations}')
    rows = []
    for note in tqdm(notes, desc=venue_id):
        content = note.content or {}
        decision, label = label_from_submission_content(content, venue_id)
        if label is None:
            decision, label = decision_from_replies(note)
        if label is None:
            continue
        pdf = extract_content_value(content, 'pdf', '')
        rows.append({
            'venue': venue_id,
            'forum': note.forum,
            'id': note.id,
            'title': extract_content_value(content, 'title', ''),
            'abstract': extract_content_value(content, 'abstract', ''),
            'keywords': extract_content_value(content, 'keywords', []),
            'decision': decision,
            'label': label,
            'pdf': pdf,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--venues', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    client = get_client()
    all_rows = []
    for venue in args.venues:
        try:
            all_rows.extend(collect_venue(client, venue))
        except Exception as e:
            print(f'WARN: failed venue {venue}: {e}')
    write_jsonl(all_rows, args.out)
    print(f'wrote {len(all_rows)} labeled rows to {args.out}')


if __name__ == '__main__':
    main()
