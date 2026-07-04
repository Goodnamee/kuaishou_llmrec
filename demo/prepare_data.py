#!/usr/bin/env python3
"""
Complete data preparation for OneReason baseline (Plan A).

Combines:
  - HF streaming: OneReason_General (general instruction data)
  - Local: competition platform JSONL (懂用户 + 懂推荐 + 懂物料)

Output: single Alpaca JSONL for LLaMA-Factory training.

Pipeline (replaces run_all.sh step 01):
  00_install.sh         -> environment (once)
  prepare_data.py        -> THIS SCRIPT: all data -> data_final.jsonl
  02_register_dataset.py -> register in LLaMA-Factory
  03_train.sh            -> train

Usage (AutoDL):
  python demo/prepare_data.py \
    --competition-data /root/autodl-tmp/data/competition/ \
    --output demo/data/data_final.jsonl \
    --general-max-samples 50000 \
    --shuffle --report
"""

import argparse, collections, json, os, random, re, sys
from pathlib import Path
import pandas as pd

# ===== Constants (same as convertv2.py) =====

_TOKENS_TO_DELETE = [
    '<|sid_end|>', '<|goods_sid_end|>', '<|living_end|>',
    '<|ad_end|>', '<|prod_end|>', '<|video_end|>',
]
_TOKENS_TO_NORMALIZE = [
    ('<|live_begin|>', '<|living_begin|>'),
    ('<prod_s_', '<s_'),
    ('<|pid_video_begin|>', '<pid_video_begin>'),
    ('<|pid_video_end|>', '<pid_video_end>'),
    ('<|pid_ad_begin|>', '<pid_ad_begin>'),
    ('<|pid_ad_end|>', '<pid_ad_end>'),
    ('<|pid_prod_begin|>', '<pid_prod_begin>'),
    ('<|pid_prod_end|>', '<pid_prod_end>'),
    ('<|pid_living_begin|>', '<pid_living_begin>'),
    ('<|pid_living_end|>', '<pid_living_end>'),
]

_HF_MIRROR = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
HF_BASE = f'{_HF_MIRROR}/datasets/OpenOneRec/Explorer_LLM_Rec_Competition/resolve/main'


# ===== Token cleaning (same logic as convertv2.py) =====

def filter_sid_end_tokens(text, stats=None):
    for tok in _TOKENS_TO_DELETE:
        if tok in text:
            cnt = text.count(tok)
            if stats is not None:
                stats[f'delete:{tok}'] += cnt
            text = text.replace(tok, '')
    for src, dst in _TOKENS_TO_NORMALIZE:
        if src in text:
            cnt = text.count(src)
            if stats is not None:
                stats[f'normalize:{src}'] += cnt
            text = text.replace(src, dst)
    return text


# ===== Message extraction (same logic as convertv2.py) =====

def extract_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, dict) and content.get('type') == 'text':
        return content['text']
    elif isinstance(content, list):
        return ''.join(
            c['text'] if isinstance(c, dict) and c.get('type') == 'text' else str(c)
            for c in content
        )
    return str(content)


def messages_to_alpaca(messages, do_filter_sid=True, add_think_pattern=True, stats=None):
    """Convert OpenAI-format messages to Alpaca record (same as convertv2.py)."""
    msg_list = []
    for msg in messages:
        role = msg['role']
        text = extract_text(msg['content'])
        if do_filter_sid:
            text = filter_sid_end_tokens(text, stats)
        msg_list.append({'role': role, 'content': text})

    # Inject /think /no_think markers
    if add_think_pattern:
        for i, msg in enumerate(msg_list):
            if msg['role'] != 'assistant':
                continue
            user_idx = i - 1
            if user_idx < 0 or msg_list[user_idx]['role'] != 'user':
                continue
            match = re.search(r'<think>(.*?)</think>', msg['content'], re.DOTALL)
            if match is None:
                msg_list[user_idx]['content'] += '/no_think'
                msg_list[i]['content'] = '<think>\n\n</think>\n' + msg['content']
                if stats is not None:
                    stats['think:inject_empty'] += 1
            elif match.group(1).strip():
                msg_list[user_idx]['content'] += '/think'
                if stats is not None:
                    stats['think:keep_existing'] += 1
            else:
                msg_list[user_idx]['content'] += '/no_think'
                if stats is not None:
                    stats['think:empty_tag'] += 1

    instruction = ''
    user_messages = []
    assistant_messages = []
    for msg in msg_list:
        if msg['role'] == 'system':
            instruction = msg['content']
        elif msg['role'] in ('user', 'human'):
            user_messages.append(msg['content'])
        elif msg['role'] == 'assistant':
            assistant_messages.append(msg['content'])

    if not user_messages or not assistant_messages:
        return None

    record = {
        'instruction': instruction,
        'input': user_messages[0],
        'output': assistant_messages[-1],
        'history': [],
    }
    if len(user_messages) > 1 or len(assistant_messages) > 1:
        n_pairs = min(len(user_messages) - 1, len(assistant_messages))
        for j in range(n_pairs):
            record['history'].append([user_messages[j], assistant_messages[j]])
    return record


# ===== Source 1: OneReason_General (HF streaming) =====

def _gen_urls(subdir, n_files):
    return [f'{HF_BASE}/data/{subdir}/part-{i:05d}.parquet' for i in range(n_files)]


def process_general(args, stats):
    """Stream OneReason_General from HF and convert to Alpaca."""
    records = []
    urls = _gen_urls('OneReason_General', 158)
    if args.general_max_samples is not None:
        random.shuffle(urls)

    n_target = args.general_max_samples
    print(f'[OneReason_General] scanning {len(urls)} files, '
          f'target ~{n_target or "all"}...', file=sys.stderr)

    for url in urls:
        if n_target and len(records) >= n_target:
            break
        try:
            df = pd.read_parquet(url)
            for _, row in df.iterrows():
                if n_target and len(records) >= n_target:
                    break
                raw = row.get('messages')
                if raw is None or isinstance(raw, float):
                    stats['skip:general_no_messages'] += 1
                    continue
                try:
                    messages = json.loads(raw) if isinstance(raw, str) else raw
                    record = messages_to_alpaca(
                        messages,
                        do_filter_sid=args.filter_sid_tokens,
                        add_think_pattern=args.add_think_pattern,
                        stats=stats,
                    )
                    if record is None:
                        stats['skip:general_empty'] += 1
                        continue
                    records.append(record)
                    stats['kept:general'] += 1
                except Exception:
                    stats['skip:general_exception'] += 1
        except Exception as e:
            print(f'  [WARN] {url}: {e}', file=sys.stderr)

    print(f'[OneReason_General] kept {len(records)} records', file=sys.stderr)
    return records


# ===== Source 2: Competition JSONL (懂用户 + 懂推荐 + 懂物料) =====

def process_competition_jsonl(args, stats):
    """Load competition JSONL files ({system, prompt, response} format)."""
    records = []
    data_dir = Path(args.competition_data)
    if not data_dir.exists():
        print(f'[WARN] Competition data dir not found: {data_dir}', file=sys.stderr)
        return records

    jsonl_files = sorted(data_dir.rglob('*.jsonl'))
    if not jsonl_files:
        print(f'[WARN] No .jsonl files found in {data_dir}', file=sys.stderr)
        return records

    print(f'[Competition] found {len(jsonl_files)} JSONL files', file=sys.stderr)
    for f in jsonl_files:
        fsize_mb = f.stat().st_size / 1024 / 1024
        print(f'  - {f.name} ({fsize_mb:.0f} MB)', file=sys.stderr)
        with open(f, 'r', encoding='utf-8') as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, list) and obj:
                    obj = obj[0]
                if not isinstance(obj, dict):
                    continue

                system = obj.get('system', '') or ''
                prompt = obj.get('prompt', '') or ''
                response = obj.get('response', '') or ''
                if not prompt and not response:
                    continue

                records.append({
                    'instruction': system,
                    'input': prompt,
                    'output': response,
                    'history': [],
                })
                stats['kept:competition'] += 1

    print(f'[Competition] kept {len(records)} records', file=sys.stderr)
    return records


# ===== Main =====

def main():
    parser = argparse.ArgumentParser(
        description='Prepare combined training data for OneReason baseline (Plan A)'
    )
    parser.add_argument('--competition-data', default=None,
                        help='Path to competition JSONL data directory')
    parser.add_argument('--no-general', action='store_true',
                        help='Skip OneReason_General')
    parser.add_argument('--general-max-samples', type=int, default=None,
                        help='Max samples from OneReason_General')
    parser.add_argument('--output', required=True,
                        help='Output Alpaca JSONL path')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle all records before writing')
    parser.add_argument('--shuffle-seed', type=int, default=2026)
    parser.add_argument('--no-filter-sid-tokens', dest='filter_sid_tokens',
                        action='store_false')
    parser.add_argument('--no-add-think-pattern', dest='add_think_pattern',
                        action='store_false')
    parser.add_argument('--report', action='store_true',
                        help='Print statistics report')
    parser.set_defaults(filter_sid_tokens=True, add_think_pattern=True)
    args = parser.parse_args()

    stats = collections.Counter()
    all_records = []

    # 1) HF OneReason_General (streaming)
    if not args.no_general:
        all_records.extend(process_general(args, stats))

    # 2) Competition JSONL (local)
    if args.competition_data:
        all_records.extend(process_competition_jsonl(args, stats))

    if not all_records:
        print('[ERROR] No records produced. Check input data.', file=sys.stderr)
        sys.exit(1)

    # Shuffle
    if args.shuffle:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(all_records)
        print(f'[INFO] shuffled {len(all_records)} records '
              f'(seed={args.shuffle_seed})', file=sys.stderr)

    # Move first record with history to front (avoid datasets null-type inference)
    first_hist = next((i for i, r in enumerate(all_records)
                       if r and r.get('history')), None)
    if first_hist is not None and first_hist > 0:
        all_records.insert(0, all_records.pop(first_hist))

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f'\n[OK] Written {len(all_records)} samples to {args.output}',
          file=sys.stderr)

    # Report
    if args.report:
        print('\n=== Statistics ===', file=sys.stderr)
        for prefix, label in [
            ('kept:', 'Kept'),
            ('skip:', 'Skipped'),
            ('delete:', 'Token Deletions'),
            ('normalize:', 'Token Normalizations'),
            ('think:', 'Think Pattern'),
        ]:
            items = [(k, v) for k, v in stats.items() if k.startswith(prefix)]
            if items:
                print(f'\n[{label}]', file=sys.stderr)
                for k, v in sorted(items):
                    print(f'  {k:<50} {v:>10,}', file=sys.stderr)


if __name__ == '__main__':
    main()
