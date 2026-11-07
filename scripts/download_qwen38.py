"""Pinned, CPU-only Qwen3.8-27B transfer; no model loading or GPU queue edits."""
import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import stat
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

from scripts import resumable_ranges
from scripts.resumable_ranges import fetch_part
from wll.storage import atomic_json

ROOT = Path('/home/hts/workflow-learning-lab')
MODEL = Path('/home/data4/hts/workflow-learning-lab/models/qwen38-27b-1d4bf0f2')
REPO = 'Qwen/Qwen3.8-27B'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
METADATA = 'artifacts/qwen38-27b-metadata-001'
PLAN = 'artifacts/qwen38-27b-download-plan-001.json'
STATE = 'artifacts/qwen38-27b-download-state-001.json'
REQUIRED = {'LICENSE', 'README.md', 'chat_template.jinja', 'config.json',
            'generation_config.json', 'merges.txt', 'model.safetensors.index.json',
            'preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json',
            'video_preprocessor_config.json', 'vocab.json'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def hex_pin(value, length):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{' + str(length) + '}', value):
        raise ValueError('invalid immutable digest or revision')
    return value


def verify_file(path, entry):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('expected regular file')
    if info.st_size != entry['size']:
        raise ValueError('file size mismatch: ' + entry['name'])
    actual = digest(path)
    if entry.get('sha256') and actual != entry['sha256']:
        raise ValueError('file digest mismatch: ' + entry['name'])
    if entry.get('git_blob'):
        blob = hashlib.sha1(f'blob {info.st_size}\0'.encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024**2), b''):
                blob.update(block)
        if blob.hexdigest() != entry['git_blob']:
            raise ValueError('HF Git blob mismatch: ' + entry['name'])
    return {'name': entry['name'], 'size': info.st_size, 'sha256': actual,
            'stat': {'size': info.st_size, 'mtime_ns': info.st_mtime_ns, 'inode': info.st_ino}}


def build_entries(upstream, mirror, seeds):
    if upstream['id'] != REPO or upstream['sha'] != REVISION or mirror['Success'] is not True:
        raise ValueError('repository identity mismatch')
    rows = upstream['siblings']
    names = [x['rfilename'] for x in rows]
    if len(set(names)) != len(names) or not REQUIRED <= set(names):
        raise ValueError('duplicate or missing upstream files')
    mirror_rows = mirror['Data']['Files']
    mirrors = {x['Name']: x for x in mirror_rows}
    if len(mirrors) != len(mirror_rows):
        raise ValueError('duplicate mirror files')
    selected = [x for x in rows if x['rfilename'] in REQUIRED or x['rfilename'].endswith('.safetensors')]
    entries = []
    for row in sorted(selected, key=lambda x: (x['rfilename'].endswith('.safetensors'), x['rfilename'])):
        name, size = row['rfilename'], row['size']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in ('.', '..'):
            raise ValueError('unsafe filename')
        if type(size) is not int or size <= 0:
            raise ValueError('invalid size')
        item = {'name': name, 'size': size}
        if 'lfs' in row:
            if row['lfs']['size'] != size:
                raise ValueError('LFS size mismatch')
            item['sha256'] = hex_pin(row['lfs']['sha256'], 64)
        else:
            if name.endswith('.safetensors'):
                raise ValueError('weights require HF SHA256')
            item['git_blob'] = hex_pin(row['blobId'], 40)
        source = seeds / name
        if source.exists() or source.is_symlink():
            item['sha256'] = verify_file(source, item)['sha256']
            item['seed'] = METADATA + '/' + name
        else:
            other = mirrors.get(name)
            if (not other or other['Size'] != size or other['Path'] != name
                    or other['Type'] != 'blob'):
                raise ValueError('mirror size/path mismatch')
            sha = hex_pin(other['Sha256'], 64)
            if 'sha256' in item and item['sha256'] != sha:
                raise ValueError('mirror differs from HF digest')
            item['sha256'] = sha
            revision = hex_pin(other['Revision'], 40)
            item['url'] = f'https://modelscope.cn/models/{REPO}/resolve/{revision}/{name}'
        entries.append(item)
    if not any(x['name'].endswith('.safetensors') for x in entries):
        raise ValueError('no weights')
    return entries


def stage_file(entry, model, parts, root, *, workers, part_bytes=128 * 1024**2):
    target = model / entry['name']
    if target.exists() or target.is_symlink():
        return verify_file(target, entry)
    temporary = model / (entry['name'] + '.assembling-001')
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError('preserved assembly exists; inspect before a new attempt')
    if 'seed' in entry:
        source = root / entry['seed']
        verify_file(source, entry)
        with source.open('rb') as src, temporary.open('xb') as out:
            shutil.copyfileobj(src, out)
    elif entry['size'] <= part_bytes:
        # Small configuration files need not support HTTP Range. The final HF
        # Git blob (or LFS hash), mirror SHA256 and exact size are all checked.
        with resumable_ranges.urlopen(entry['url'], timeout=90) as response:
            if response.status != 200:
                raise ValueError('unexpected full-file HTTP status')
            data = response.read(entry['size'] + 1)
        with temporary.open('xb') as out:
            out.write(data)
    else:
        jobs = [(entry['url'], start, min(start + part_bytes, entry['size']) - 1,
                 entry['size'], parts / f'{entry["name"]}.{start}-{min(start + part_bytes, entry["size"]) - 1}')
                for start in range(0, entry['size'], part_bytes)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch_part, job, request_bytes=16 * 1024**2,
                                   max_failures=8) for job in jobs]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        with temporary.open('xb') as out:
            for job in jobs:
                with job[-1].open('rb') as src:
                    shutil.copyfileobj(src, out, 4 * 1024**2)
    verify_file(temporary, entry)
    # Link fails if a final file appeared concurrently, instead of overwriting it.
    os.link(temporary, target)
    temporary.unlink()
    return verify_file(target, entry)


def load_plan(expected):
    if digest(ROOT / PLAN) != expected:
        raise ValueError('plan digest mismatch')
    plan = json.loads((ROOT / PLAN).read_text())
    if (socket.gethostname() != 'amax' or os.getuid() != 1031
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or Path.cwd() != ROOT or plan['model_directory'] != str(MODEL)):
        raise ValueError('wrong host, identity, working directory or CPU-only environment')
    for name, sha in plan['source_sha256'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('source digest mismatch: ' + name)
    upstream = json.loads((ROOT / METADATA / 'repository.json').read_text())
    mirror = json.loads((ROOT / 'artifacts/qwen38-27b-modelscope-metadata-001.json').read_text())
    entries = build_entries(upstream, mirror, ROOT / METADATA)
    if plan['files'] != entries or plan['total_bytes'] != sum(x['size'] for x in entries):
        raise ValueError('catalog differs from registered plan')
    index = json.loads((ROOT / METADATA / 'model.safetensors.index.json').read_text())
    if set(index['weight_map'].values()) != {x['name'] for x in entries if x['name'].endswith('.safetensors')}:
        raise ValueError('weight index coverage mismatch')
    if plan['workers'] != 4 or plan['max_download_seconds'] != 21600:
        raise ValueError('unregistered resource settings')
    if MODEL.parent.resolve() != MODEL.parent or MODEL.parent.stat().st_uid != os.getuid():
        raise ValueError('unexpected bulk parent')
    if MODEL.is_symlink() or (MODEL.exists() and MODEL.stat().st_uid != os.getuid()):
        raise ValueError('unexpected model directory')
    if shutil.disk_usage(MODEL.parent).free < 2 * plan['total_bytes'] + 10 * 1024**3:
        raise OSError('insufficient bulk disk space including retained parts')
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    plan = load_plan(args.plan_sha256)
    if args.check:
        print(json.dumps({'preflight': 'passed', 'files': len(plan['files']),
                          'total_bytes': plan['total_bytes'], 'uses_gpu': False}))
        return
    resumable_ranges.urlopen = build_opener(ProxyHandler({})).open
    with (ROOT / 'artifacts/qwen38-27b-download.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / STATE).exists():
            raise FileExistsError('attempt exists; no automatic restart')
        MODEL.mkdir(exist_ok=True)
        parts = MODEL / 'verified-range-parts-001'
        if parts.is_symlink():
            raise ValueError('unexpected parts directory')
        parts.mkdir(exist_ok=True)
        state = {'phase': 'downloading', 'pid': os.getpid(), 'uid': os.getuid(),
                 'start_ticks': Path('/proc/self/stat').read_text().split(')', 1)[1].split()[19],
                 'started_epoch': time.time(), 'plan_sha256': args.plan_sha256,
                 'uses_gpu': False, 'verified': [], 'total_bytes': plan['total_bytes'],
                 'initial_part_bytes': sum(p.stat().st_size for p in parts.iterdir())}
        atomic_json(ROOT / STATE, state)
        try:
            for entry in plan['files']:
                state.update(filename=entry['name'], updated_epoch=time.time())
                atomic_json(ROOT / STATE, state)
                record = stage_file(entry, MODEL, parts, ROOT, workers=plan['workers'])
                state['verified'].append(record)
                state['verified_bytes'] = sum(x['size'] for x in state['verified'])
                atomic_json(ROOT / STATE, state)
                print(json.dumps({'verified': entry['name'], 'verified_bytes': state['verified_bytes']}), flush=True)
            atomic_json(MODEL / 'weight-source.json', {
                'hf_repo': REPO, 'hf_revision': REVISION, 'plan_sha256': args.plan_sha256,
                'files': state['verified'], 'finished_epoch': time.time()})
            state.update(phase='complete', finished_epoch=time.time())
        except BaseException as exc:
            state.update(phase='failed', error_type=type(exc).__name__, finished_epoch=time.time())
            atomic_json(ROOT / STATE, state)
            raise
        atomic_json(ROOT / STATE, state)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
