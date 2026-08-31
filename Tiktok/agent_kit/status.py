"""Machine-readable run status, written continuously for the live dashboard.

The agent's own stdout is a scrolling narrative: fine for a post-mortem, useless
for answering "what is it doing right now, and is it winning?" while a six-hour
run is in flight. This module is the other half -- a single small JSON file that
always holds the CURRENT state, overwritten in place rather than appended.

Design notes, both learned the hard way on Windows:

  * Writes are atomic (tmp file + os.replace). The dashboard polls this file
    several times a second, and a plain open('w') leaves a window in which the
    reader sees a truncated file and a JSONDecodeError.
  * Nothing here may raise into the training loop. A dashboard that breaks is an
    annoyance; a dashboard that kills a five-hour run is a disaster. Every public
    function swallows its own exceptions.

The writer is the agent process; the reader is dashboard.py. There is exactly
one writer, so no locking beyond a thread lock is needed.
"""
import json
import os
import threading
import time

RUNS_DIR = os.environ.get('AGENT_RUNS_DIR', './runs')
STATUS_PATH = os.path.join(RUNS_DIR, 'status.json')

MAX_EVENTS = 40
MAX_EPOCH_HIST = 60

_LOCK = threading.Lock()
_STATE = {
    'pid': os.getpid(),
    'started': time.time(),
    'updated': time.time(),
    'phase': 'starting',
    'phase_detail': '',
    'iteration': 0,
    'max_iterations': 0,
    'elapsed_s': 0.0,
    'budget_s': 0.0,
    'current': {},
    'best': {},
    'ensemble': {},
    'final': {},
    'tokens': {'in': 0, 'out': 0, 'cost_usd': 0.0},
    'counts': {'ok': 0, 'timeout': 0, 'failed': 0, 'note': 0},
    'events': [],
    'alive': True,
}


def _flush():
    """Write the whole state atomically. Caller holds the lock."""
    try:
        _STATE['updated'] = time.time()
        os.makedirs(os.path.dirname(STATUS_PATH) or '.', exist_ok=True)
        tmp = STATUS_PATH + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(_STATE, fh, default=str)
        os.replace(tmp, STATUS_PATH)
    except Exception:                                             # noqa: BLE001
        pass                       # never let the dashboard break the run


def update(**kw):
    """Merge top-level keys and flush."""
    try:
        with _LOCK:
            _STATE.update(kw)
            _flush()
    except Exception:                                             # noqa: BLE001
        pass


def patch(section, **kw):
    """Merge keys into one nested dict (e.g. patch('current', epoch=3))."""
    try:
        with _LOCK:
            cur = _STATE.setdefault(section, {})
            if not isinstance(cur, dict):
                cur = _STATE[section] = {}
            cur.update(kw)
            _flush()
    except Exception:                                             # noqa: BLE001
        pass


def set_section(section, value):
    try:
        with _LOCK:
            _STATE[section] = value
            _flush()
    except Exception:                                             # noqa: BLE001
        pass


def event(kind, text):
    """Append to the ring buffer of notable moments.

    kind drives the dashboard's colour: 'best' | 'ok' | 'warn' | 'fail' |
    'agent' | 'tool' | 'info'.
    """
    try:
        with _LOCK:
            evs = _STATE.setdefault('events', [])
            evs.append({'t': time.time(), 'kind': kind, 'text': str(text)[:400]})
            del evs[:-MAX_EVENTS]
            _flush()
    except Exception:                                             # noqa: BLE001
        pass


def push_epoch(value):
    """Append one epoch's validation primary to the live curve."""
    try:
        with _LOCK:
            cur = _STATE.setdefault('current', {})
            hist = cur.setdefault('epoch_hist', [])
            hist.append(round(float(value), 6))
            del hist[:-MAX_EPOCH_HIST]
            _flush()
    except Exception:                                             # noqa: BLE001
        pass


def bump(counter):
    try:
        with _LOCK:
            c = _STATE.setdefault('counts', {})
            c[counter] = c.get(counter, 0) + 1
            _flush()
    except Exception:                                             # noqa: BLE001
        pass


def finish(ok=True, note=''):
    update(phase='done' if ok else 'stopped', phase_detail=note, alive=False)


def read(path=None):
    """Reader side, used by the dashboard. Returns {} if unreadable."""
    try:
        with open(path or STATUS_PATH, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:                                             # noqa: BLE001
        return {}
