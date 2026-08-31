# -*- coding: utf-8 -*-
"""Live terminal dashboard for the autonomous research agent.

Run it in a second PowerShell window (or let run.ps1 open it) while the agent
works:

    python dashboard.py

It is a strict READER. It never imports the harness, never touches the journal
for writing, and holds no lock -- the agent's run must not be able to fail
because the dashboard did. Everything it shows comes from two files the agent
writes as it goes: runs/status.json (current state, overwritten atomically) and
runs/journal.jsonl (append-only history).

Design constraints that shaped this file, all of them Windows-specific:

  * PowerShell's default console (conhost, WindowsPowerShell 5.1) does not
    interpret ANSI escapes until a process asks it to, so we flip
    ENABLE_VIRTUAL_TERMINAL_PROCESSING via ctypes at start-up. Without it the
    screen fills with literal escape codes.
  * The console codepage is often cp1252, which cannot encode the block
    characters used for bars and sparklines. We probe the encoder once and fall
    back to an all-ASCII glyph set rather than crashing on the first redraw.
  * Redrawing with cls/clear flickers badly at 2 Hz. Instead the cursor is
    homed and every line is written with an erase-to-end-of-line, so unchanged
    pixels are never blanked.

Keys:  q quit (the agent keeps running)   p pause redraw
"""
import io
import json
import os
import shutil
import sys
import time

RUNS_DIR = os.environ.get('AGENT_RUNS_DIR', './runs')
STATUS_PATH = os.path.join(RUNS_DIR, 'status.json')
JOURNAL_PATH = os.path.join(RUNS_DIR, 'journal.jsonl')

FM_VALID_PRIMARY = 0.6016              # mirrored from agent_kit/journal.py
ORACLE_VALID_PRIMARY = 0.8484
RANDOM_VALID_PRIMARY = 0.4834

STALE_AFTER_S = 90.0                   # no status write for this long -> warn


# ---------------------------------------------------------------- terminal ---
def enable_vt():
    """Ask the Windows console to interpret ANSI escapes. No-op elsewhere."""
    if os.name != 'nt':
        return True
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)                     # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not k.GetConsoleMode(h, ctypes.byref(mode)):
            return False
        return bool(k.SetConsoleMode(h, mode.value | 0x0004))
    except Exception:                                             # noqa: BLE001
        return False


def _can_encode(sample):
    enc = getattr(sys.stdout, 'encoding', None) or 'ascii'
    try:
        sample.encode(enc)
        return True
    except Exception:                                             # noqa: BLE001
        return False


for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

VT = enable_vt()
UNI = _can_encode('─█▁')             # box + full block + low block

if UNI:
    G = {'h': '─', 'v': '│', 'tl': '┌', 'tr': '┐',
         'bl': '└', 'br': '┘', 'ml': '├', 'mr': '┤',
         'full': '█', 'empty': '░', 'dot': '·',
         'spark': '▁▂▃▄▅▆▇█',
         'ok': '✔', 'bad': '✘', 'warn': '!', 'run': '●'}
else:
    G = {'h': '-', 'v': '|', 'tl': '+', 'tr': '+', 'bl': '+', 'br': '+',
         'ml': '+', 'mr': '+', 'full': '#', 'empty': '.', 'dot': '.',
         'spark': '.:-=+*#%@', 'ok': 'v', 'bad': 'x', 'warn': '!', 'run': 'o'}

RESET, BOLD, DIM = '\033[0m', '\033[1m', '\033[2m'
RED, GREEN, YELLOW, BLUE, CYAN, GREY = ('\033[31m', '\033[32m', '\033[33m',
                                        '\033[34m', '\033[36m', '\033[90m')
NO_COLOR = os.environ.get('NO_COLOR') is not None or not VT


def c(colour, text):
    return text if NO_COLOR else '%s%s%s' % (colour, text, RESET)


def visible_len(s):
    """Length ignoring ANSI sequences -- padding must not count escape bytes."""
    out, i = 0, 0
    while i < len(s):
        if s[i] == '\033':
            j = s.find('m', i)
            i = len(s) if j < 0 else j + 1
            continue
        out += 1
        i += 1
    return out


def clip(s, width):
    """Truncate to `width` visible columns, keeping escapes balanced."""
    if visible_len(s) <= width:
        return s
    out, seen, i = [], 0, 0
    while i < len(s) and seen < width:
        if s[i] == '\033':
            j = s.find('m', i)
            if j < 0:
                break
            out.append(s[i:j + 1])
            i = j + 1
            continue
        out.append(s[i])
        seen += 1
        i += 1
    out.append(RESET if not NO_COLOR else '')
    return ''.join(out)


# ------------------------------------------------------------------ widgets --
def bar(frac, width, colour=GREEN):
    frac = 0.0 if frac is None else max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return c(colour, G['full'] * n) + c(GREY, G['empty'] * (width - n))


def headroom_bar(primary, width=34):
    """Where a score sits from a random scorer to the oracle, FM marked.

    Spanning random->oracle rather than FM->oracle on purpose: every result this
    benchmark can realistically produce sits in the first 2% of the FM->oracle
    span, so that bar would always look empty and tell the operator nothing.
    """
    lo, hi = RANDOM_VALID_PRIMARY, ORACLE_VALID_PRIMARY
    pos = int(round(max(0.0, min(1.0, (primary - lo) / (hi - lo))) * width))
    fm = int(round((FM_VALID_PRIMARY - lo) / (hi - lo) * width))
    cells = []
    for i in range(width):
        if i == fm:
            cells.append(c(YELLOW + BOLD, G['v']))
        elif i < pos:
            cells.append(c(GREEN, G['full']))
        else:
            cells.append(c(GREY, G['empty']))
    return 'rnd[' + ''.join(cells) + ']oracle'


def sparkline(values, width=None):
    if not values:
        return ''
    vals = values[-width:] if width else values
    lo, hi = min(vals), max(vals)
    ramp = G['spark']
    if hi - lo < 1e-9:
        return ramp[len(ramp) // 2] * len(vals)
    span = len(ramp) - 1
    return ''.join(ramp[int(round((v - lo) / (hi - lo) * span))] for v in vals)


def plot(points, width, height=6):
    """Tiny column chart of validation primary, with the FM baseline drawn in.

    Rendered as text rows rather than a sparkline because the question the
    operator actually asks -- "are we above the baseline and still climbing?" --
    needs the FM line visible, and a one-row sparkline cannot show it.
    """
    if not points:
        return [c(GREY, '(no scored experiments yet)')]
    vals = [p['valid'] for p in points][-width:]
    lo = min(min(vals), FM_VALID_PRIMARY) - 0.0004
    hi = max(max(vals), FM_VALID_PRIMARY) + 0.0004
    span = max(hi - lo, 1e-9)
    fm_row = int(round((1 - (FM_VALID_PRIMARY - lo) / span) * (height - 1)))
    best = max(vals)

    rows = []
    for r in range(height):
        cells = []
        for v in vals:
            vr = int(round((1 - (v - lo) / span) * (height - 1)))
            if vr == r:                                   # the value itself
                cells.append(c(GREEN + BOLD if v == best else GREEN, G['full']))
            elif vr < r:                                  # column body below it
                cells.append(c(YELLOW if r == fm_row else GREY, G['empty']))
            elif r == fm_row:                             # baseline through empty space
                cells.append(c(YELLOW, G['h']))
            else:
                cells.append(' ')
        axis = '%.4f' % (hi - span * r / max(1, height - 1))
        label = c(GREY, axis) + c(GREY, ' ' + G['v'] + ' ')
        if r == fm_row:
            label = c(YELLOW, axis) + c(GREY, ' ' + G['v'] + ' ')
        rows.append(label + ''.join(cells))
    rows.append(c(GREY, ' ' * 6 + ' ' + G['v'] + ' ') +
                c(GREY, ('%s FM baseline %.4f  %s best %.4f  (%d experiments)'
                         % (G['h'], FM_VALID_PRIMARY, G['full'], best, len(points)))))
    return rows


def process_alive(pid):
    """Is that PID still running? Windows-only, read-only, never raises.

    status.json cannot report its own hard kill -- a taskkill or a stray Ctrl-C
    leaves the last frame saying alive=true forever. Checking the PID is the
    only way to tell "stalled, still thinking" apart from "gone", and those two
    need very different reactions from whoever is watching.
    """
    if not pid or os.name != 'nt':
        return None                     # unknown; caller falls back to staleness
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))   # QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_uint32()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259         # STILL_ACTIVE
    except Exception:                                             # noqa: BLE001
        return None


def describe(cfg):
    """One-line config summary. Mirrors agent_kit/progress.py's describe().

    Duplicated rather than imported on purpose: the dashboard must never import
    the agent's package, because that would drag in torch and make the viewer
    fail whenever the training environment is mid-edit.
    """
    if not isinstance(cfg, dict):
        return '' if cfg is None else str(cfg)
    m, l, t = cfg.get('model') or {}, cfg.get('loss') or {}, cfg.get('train') or {}
    bits = ['%s/%s' % (m.get('type', 'fm'), l.get('type', 'bce'))]
    for src, keys in ((m, ('k', 'p', 'L', 'hidden')),
                      (t, ('lr', 'l2', 'batch', 'group_size'))):
        for k in keys:
            if k in src:
                bits.append('%s=%s' % (k, src[k]))
    return ' '.join(bits)


def hhmm(seconds):
    if seconds is None:
        return '--:--'
    seconds = int(max(0, seconds))
    h, m = divmod(seconds // 60, 60)
    return '%dh%02dm' % (h, m) if h else '%dm%02ds' % (m, seconds % 60)


def wrap(text, width, limit):
    words, lines, cur = str(text or '').split(), [], ''
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
            if len(lines) >= limit:
                break
        else:
            cur = (cur + ' ' + w).strip()
    if cur and len(lines) < limit:
        lines.append(cur)
    return lines or ['']


# ------------------------------------------------------------------- input ---
def read_key():
    """Non-blocking single keypress, or None. Windows console only."""
    try:
        import msvcrt
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            return ch.decode('latin-1', 'ignore').lower()
    except Exception:                                             # noqa: BLE001
        pass
    return None


# -------------------------------------------------------------------- data ---
def read_status():
    try:
        with io.open(STATUS_PATH, encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:                                             # noqa: BLE001
        return {}


def read_history(limit=200):
    """Scored experiments straight from the journal.

    The journal is the authority even though status.json carries a copy: the
    dashboard is often started mid-run, and the copy in status only covers what
    happened since the agent process came up.
    """
    out = []
    try:
        with io.open(JOURNAL_PATH, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get('valid_primary') is not None:
                    out.append({'iteration': e.get('iteration'),
                                'valid': e['valid_primary'],
                                'status': e.get('status'),
                                'GAUC': e.get('valid_GAUC'),
                                'ndcg': e.get('valid_nDCG@5'),
                                'unbiased': e.get('unbiased_primary'),
                                'config': e.get('config')})
    except Exception:                                             # noqa: BLE001
        return []
    return out[-limit:]


PHASE_STYLE = {
    'training': (GREEN, 'training a model'),
    'thinking': (CYAN, 'agent is choosing the next move'),
    'tuning': (BLUE, 'Bayesian hyper-parameter search'),
    'ensembling': (BLUE, 'portfolio search + ensemble selection'),
    'reflecting': (CYAN, 'reading the result'),
    'finalizing': (YELLOW, 'designating the final submission'),
    'submitting': (YELLOW, 'building submission.csv'),
    'starting': (GREY, 'starting up'),
    'done': (GREEN, 'run complete'),
    'stopped': (RED, 'stopped'),
}

EVENT_STYLE = {'best': (GREEN + BOLD, '*'), 'ok': (GREEN, G['ok']),
               'warn': (YELLOW, G['warn']), 'fail': (RED, G['bad']),
               'agent': (CYAN, '>'), 'tool': (BLUE, '+'), 'info': (GREY, '-')}


# ------------------------------------------------------------------ render ---
def render(st, hist, width, height):
    """Build the whole frame as a list of lines. Pure function, easy to eyeball."""
    inner = width - 2
    L = []

    def rule(title=''):
        if not title:
            return c(GREY, G['ml'] + G['h'] * inner + G['mr'])
        t = ' %s ' % title
        return c(GREY, G['ml'] + G['h'] + c(BOLD, t) + c(GREY, G['h'] * max(
            0, inner - len(t) - 1)) + G['mr'])

    def row(s=''):
        return c(GREY, G['v']) + ' ' + clip(s, inner - 2).ljust(
            inner - 2 + (len(clip(s, inner - 2)) - visible_len(clip(s, inner - 2)))) \
            + ' ' + c(GREY, G['v'])

    now = time.time()
    updated = st.get('updated') or 0
    stale = st and (now - updated) > STALE_AFTER_S
    gone = bool(st.get('alive')) and process_alive(st.get('pid')) is False
    alive = bool(st.get('alive')) and not stale and not gone

    # ---- header
    title = ' AUTONOMOUS ML RESEARCH AGENT %s KuaiRand-Pure ' % G['dot']
    clock = ' %s %s up %s ' % (time.strftime('%H:%M:%S'), G['dot'],
                               hhmm(now - (st.get('started') or now)))
    pad = max(0, inner - len(title) - len(clock))
    L.append(c(GREY, G['tl']) + c(BOLD + CYAN, title) + c(GREY, G['h'] * pad)
             + c(GREY, clock) + c(GREY, G['tr']))

    # ---- phase
    phase = st.get('phase', 'unknown')
    pcol, pdesc = PHASE_STYLE.get(phase, (GREY, ''))
    if not st:
        state = c(RED, 'no status file -- the agent has not started yet')
    elif gone:
        state = c(RED + BOLD, '%s KILLED' % G['bad']) + c(
            RED, ' %s process %s is gone; the run stopped at %s mid-%s'
            % (G['dot'], st.get('pid'),
               time.strftime('%H:%M:%S', time.localtime(updated)), phase))
    elif stale:
        state = c(RED, '%s STALLED %s no update for %s'
                  % (G['bad'], G['dot'], hhmm(now - updated)))
    elif not st.get('alive'):
        state = c(GREY, '%s %s %s' % (G['ok'], phase, st.get('phase_detail') or ''))
    else:
        state = c(pcol, '%s %-11s' % (G['run'], phase)) + c(GREY, ' %s %s' % (G['dot'], pdesc))
    detail = st.get('phase_detail') or ''
    left = st.get('directive_turns_left')
    directive = (c(YELLOW, '  [directive: %d turns left]' % left)
                 if alive and left else '')
    L.append(row(c(BOLD, 'PHASE   ') + state
                 + (c(GREY, '  ' + str(detail)[:40]) if alive and detail else '')
                 + directive))

    # ---- best
    # The agent publishes `best` when it scores an experiment, but a tool can
    # hold the turn for an hour and the journal is already ahead of it. Prefer
    # whichever is higher rather than showing a stale number.
    best = dict(st.get('best') or {})
    jbest = max(hist, key=lambda e: e['valid']) if hist else None
    if jbest and (best.get('valid') is None or jbest['valid'] > best['valid']):
        best = {'iteration': jbest['iteration'], 'valid': jbest['valid'],
                'delta': jbest['valid'] - FM_VALID_PRIMARY,
                'GAUC': jbest.get('GAUC'), 'ndcg': jbest.get('ndcg'),
                'unbiased': jbest.get('unbiased'),
                'config': describe(jbest.get('config'))}
    if best.get('valid') is not None:
        d = best.get('delta') or (best['valid'] - FM_VALID_PRIMARY)
        L.append(row(c(BOLD, 'BEST    ')
                     + c(BOLD + GREEN if d > 0 else RED, '%.4f' % best['valid'])
                     + c(GREEN if d > 0 else RED, '  %+.4f vs FM' % d)
                     + c(GREY, '   iter %s %s GAUC %.4f %s nDCG@5 %.4f %s unbiased %.4f'
                         % (best.get('iteration', '?'), G['dot'],
                            best.get('GAUC') or 0, G['dot'], best.get('ndcg') or 0,
                            G['dot'], best.get('unbiased') or 0))))
        L.append(row('        ' + headroom_bar(best['valid'])
                     + c(GREY, '  %+.2f%% of FM->oracle'
                         % (100 * (best['valid'] - FM_VALID_PRIMARY)
                            / (ORACLE_VALID_PRIMARY - FM_VALID_PRIMARY)))))
        L.append(row(c(GREY, '        ' + str(best.get('config', ''))[:inner - 12])))
    else:
        L.append(row(c(GREY, 'BEST     no successful experiment yet')))

    # ---- budgets
    it, mx = st.get('iteration') or 0, st.get('max_iterations') or 50
    # Tick the clock from the start timestamp while the run is live; the
    # published elapsed_s only advances when the agent next writes status.
    # Live: tick from the start stamp. Dead: how far it actually got, which is
    # the last status write -- elapsed_s alone reads 0 for a run killed inside
    # its first turn, before any tool published a figure.
    started = st.get('started')
    if alive and started:
        el = now - started
    elif started and updated:
        el = max(st.get('elapsed_s') or 0, updated - started)
    else:
        el = st.get('elapsed_s') or 0
    bud = st.get('budget_s') or 1
    L.append(row(c(BOLD, 'BUDGET  ') + 'iters ' + bar(it / max(1, mx), 16)
                 + ' %d/%d' % (it, mx)
                 + c(GREY, '   clock ') + bar(el / max(1.0, bud), 16, CYAN)
                 + ' %s/%s' % (hhmm(el), hhmm(bud))))

    tok, cnt = st.get('tokens') or {}, st.get('counts') or {}
    L.append(row(c(BOLD, 'COST    ')
                 + c(GREY, '%s tok in %s %s out %s $%.2f'
                     % ('{:,}'.format(tok.get('in', 0)), G['dot'],
                        '{:,}'.format(tok.get('out', 0)), G['dot'],
                        tok.get('cost_usd', 0.0)))
                 + '   ' + c(GREEN, '%s %d ok' % (G['ok'], cnt.get('ok', 0)))
                 + c(YELLOW, '  %s %d timeout' % (G['warn'], cnt.get('timeout', 0)))
                 + c(RED, '  %s %d failed' % (G['bad'], cnt.get('failed', 0)))))

    # ---- current experiment
    cur = st.get('current') or {}
    if cur and alive:
        L.append(rule('CURRENT ' + ('EXPERIMENT %s' % cur.get('label', ''))[:40]))
        for i, ln in enumerate(wrap(cur.get('hypothesis'), inner - 10, 2)):
            L.append(row(c(GREY, 'hyp  ' if i == 0 else '     ') + ln))
        L.append(row(c(GREY, 'cfg  ') + str(cur.get('config', ''))))
        ep, eps = cur.get('epoch') or 0, cur.get('epochs') or 40
        seeds = cur.get('seeds') or []
        seed = cur.get('seed')
        v = cur.get('valid')
        dv = (v - FM_VALID_PRIMARY) if v is not None else None
        line = (c(GREY, 'seed ') + '%s/%d' % (('%d' % seed) if seed is not None else '-',
                                              len(seeds) or 1)
                + c(GREY, '  epoch ') + bar(ep / max(1, eps), 12, CYAN) + ' %2d/%d' % (ep, eps))
        if v is not None:
            line += ('   ' + c(BOLD, 'valid %.4f' % v)
                     + c(GREEN if dv > 0 else RED, ' %+.4f' % dv))
        if cur.get('secs_per_epoch'):
            line += c(GREY, '   %.0fs/epoch' % cur['secs_per_epoch'])
        started = cur.get('started')
        if started:
            line += c(GREY, '   %s elapsed' % hhmm(now - started))
        L.append(row(line))
        spark = sparkline(cur.get('epoch_hist') or [], inner - 14)
        if spark:
            L.append(row(c(GREY, 'curve') + ' ' + c(CYAN, spark)))

    # ---- ensemble / final
    ens, fin = st.get('ensemble') or {}, st.get('final') or {}
    if ens.get('valid') is not None:
        L.append(rule('PORTFOLIO'))
        L.append(row(c(BOLD, 'ensemble %.4f' % ens['valid'])
                     + c(GREEN if (ens.get('delta') or 0) > 0 else RED,
                         '  %+.4f vs FM' % (ens.get('delta') or 0))
                     + c(GREY, '   best single %.4f %s %d models %s %d members %s unbiased %.4f'
                         % (ens.get('best_single') or 0, G['dot'],
                            ens.get('n_models') or 0, G['dot'],
                            ens.get('n_members') or 0, G['dot'],
                            ens.get('unbiased') or 0))))
    if fin:
        L.append(rule('FINAL SUBMISSION'))
        ok = fin.get('ok')
        L.append(row(c(GREEN + BOLD if ok else RED,
                       '%s %s' % (G['ok'] if ok else G['bad'],
                                  'submission.csv written' if ok else 'FAILED'))
                     + c(GREY, '   choice=%s   %s' % (fin.get('choice'),
                                                      fin.get('designated_at', '')))))
        for ln in wrap(fin.get('reason'), inner - 8, 2):
            L.append(row(c(GREY, '  ' + ln)))

    # ---- history plot
    L.append(rule('SCORE HISTORY'))
    for ln in plot(hist, max(20, inner - 12), height=6):
        L.append(row(ln))

    # ---- agent voice
    L.append(rule('AGENT'))
    txt = st.get('agent_text') or '(waiting for the agent to speak)'
    for ln in wrap(txt, inner - 4, 4):
        L.append(row(c(CYAN, ln)))

    # ---- events, filling whatever vertical room is left
    L.append(rule('EVENTS'))
    room = max(3, height - len(L) - 2)
    for e in (st.get('events') or [])[-room:]:
        col, mark = EVENT_STYLE.get(e.get('kind'), (GREY, '-'))
        L.append(row(c(GREY, time.strftime('%H:%M:%S', time.localtime(e.get('t', now))))
                     + ' ' + c(col, mark) + ' ' + c(col, str(e.get('text', '')))))

    L.append(c(GREY, G['bl'] + G['h'] * inner + G['br']))
    L.append(c(GREY, '  q quit (agent keeps running)   p pause   %s   %s'
               % ('paused' if PAUSED[0] else 'live',
                  'status %s ago' % hhmm(now - updated) if updated else 'no status yet')))
    return L


PAUSED = [False]


def main():
    interval = 0.5
    if '--once' in sys.argv:                       # smoke test / screenshot mode
        st, hist = read_status(), read_history()
        w, h = shutil.get_terminal_size((100, 40))
        print('\n'.join(render(st, hist, max(80, min(w, 140)), max(30, h - 1))))
        return
    sys.stdout.write('\033[?25l')                  # hide cursor
    try:
        while True:
            key = read_key()
            if key == 'q':
                break
            if key == 'p':
                PAUSED[0] = not PAUSED[0]
            if not PAUSED[0]:
                w, h = shutil.get_terminal_size((100, 40))
                w, h = max(80, min(w, 160)), max(28, h)
                lines = render(read_status(), read_history(), w, h - 1)[:h - 1]
                buf = ['\033[H']
                for ln in lines:
                    buf.append(ln + '\033[K\n')
                buf.append('\033[J')               # clear anything below
                sys.stdout.write(''.join(buf))
                sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write('\033[?25h\n')            # show cursor again
        sys.stdout.flush()


if __name__ == '__main__':
    main()
