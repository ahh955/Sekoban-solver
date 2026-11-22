// Sokoban solver: split the level into its rooms, then a push-based A* per room
// (Hungarian assignment heuristic over BFS distances, dead-square + frozen 2x2
// pruning, player-region state canonicalisation). Push-optimal per room, so the
// level total is P*. Deterministic: fixed orders, budgets by expansions, no clock.
import {mkdirSync, readdirSync, readFileSync, writeFileSync} from 'node:fs';
import {join} from 'node:path';
const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const DIRS = [[0, -1, 'u', 'U'], [0, 1, 'd', 'D'], [-1, 0, 'l', 'L'], [1, 0, 'r', 'R']];
const AB = 200000, BB = 250000; // expansion budgets: A*, then BFS fallback (per room)
const WB = 20000, TOTMAX = 600000; // whole-level fallback budget; global expansion cap for the run
let TOT = 0; // expansions so far across all searches: a pathological level must fail, never time the set out

function parse(text) {
  const rows = text.split('\n').filter(r => r.length);
  const h = rows.length, w = Math.max(...rows.map(r => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals = [], boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const c = row[x], i = y * w + x;
      if (c === '#') continue;
      wall[i] = 0;
      if ('.*+'.includes(c)) goals.push(i);
      if ('$*'.includes(c)) boxes.push(i);
      if ('@+'.includes(c)) player = i;
    }
  });
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
    if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
  return {w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player};
}

// Squares from which a lone box can still reach a goal (pull back from the goals).
function liveSq(w, wall, goals) {
  const live = new Uint8Array(wall.length), q = [];
  for (const g of goals) { live[g] = 1; q.push(g); }
  for (let i = 0; i < q.length; i++) for (const [dx, dy] of DIRS) {
    const d = dy * w + dx, y = q[i] - d;
    if (!wall[y] && !wall[y - d] && !live[y]) { live[y] = 1; q.push(y); }
  }
  return live;
}

function distFrom(w, wall, g) {
  const dist = new Int32Array(wall.length).fill(-1), q = [g];
  dist[g] = 0;
  for (let i = 0; i < q.length; i++) for (const [dx, dy] of DIRS) {
    const n = q[i] + dy * w + dx;
    if (!wall[n] && dist[n] < 0) { dist[n] = dist[q[i]] + 1; q.push(n); }
  }
  return dist;
}

// Admissible push lower bound: min-cost box->goal assignment over BFS distances
// (brute-force permutations up to 4 boxes; nearest-goal sum beyond that).
function heur(boxes, dists) {
  const n = boxes.length;
  if (n === 0) return 0;
  if (n > 4) {
    let s = 0;
    for (const b of boxes) {
      let m = Infinity;
      for (const d of dists) if (d[b] >= 0 && d[b] < m) m = d[b];
      if (m === Infinity) return Infinity;
      s += m;
    }
    return s;
  }
  const used = new Array(n).fill(false);
  let best = Infinity;
  const rec = (i, sum) => {
    if (sum >= best) return;
    if (i === n) { best = sum; return; }
    for (let g = 0; g < n; g++) {
      if (used[g]) continue;
      const d = dists[g][boxes[i]];
      if (d < 0) continue;
      used[g] = true; rec(i + 1, sum + d); used[g] = false;
    }
  };
  rec(0, 0);
  return best;
}

// Lowest square of the player's reachable region (boxes as walls); marks seen.
function regionLow(w, wall, occ, from, seen, stamp, stack) {
  let low = from, n = 0;
  stack[n++] = from; seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (const [dx, dy] of DIRS) {
      const q = p + dy * w + dx;
      if (!wall[q] && !occ[q] && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; }
    }
  }
  return low;
}

// A box in a 2x2 block of walls/boxes with a box off its goal can never move out.
function frozen(w, wall, occ, goal, t) {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1, c = a + w, d = c + 1;
    if (!(wall[a] || occ[a]) || !(wall[b] || occ[b]) || !(wall[c] || occ[c]) || !(wall[d] || occ[d])) continue;
    if ((occ[a] && !goal[a]) || (occ[b] && !goal[b]) || (occ[c] && !goal[c]) || (occ[d] && !goal[d])) return true;
  }
  return false;
}

function walkTo(w, wall, occ, from, to) {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1), how = new Array(wall.length), q = [from];
  prev[from] = from;
  for (let i = 0; i < q.length; i++) {
    const p = q[i];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const n = p + dy * w + dx;
      if (wall[n] || occ[n] || prev[n] !== -1) continue;
      prev[n] = p; how[n] = letter; q.push(n);
    }
  }
  if (prev[to] === -1) throw new Error('walk unreachable');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

// Push-optimal search over (sorted boxes, player region). useH: A* with the
// Hungarian heuristic; else plain BFS (the fallback). States re-open on a
// strictly better g, so the first goal popped is optimal either way.
function search(sub, useH, budget) {
  const {w, wall, goals, boxes: sb, player: sp} = sub;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSq(w, wall, goals);
  if (sb.some(b => !live[b])) return null;
  const dists = goals.map(g => distFrom(w, wall, g));
  const occ = new Uint8Array(size), seen = new Int32Array(size), mark = new Int32Array(size), stack = new Int32Array(size);
  let stamp = 0;
  const solved = bs => bs.every(b => goal[b]);
  const keyOf = (bs, low) => low + '|' + bs.join(',');
  const bxs = [sb.slice()], ply = [sp], par = [-1], frm = [-1], tto = [-1], gs = [0];
  for (const b of sb) occ[b] = 1;
  const idx = new Map([[keyOf(sb, regionLow(w, wall, occ, sp, mark, ++stamp, stack)), 0]]);
  for (const b of sb) occ[b] = 0;
  const h0 = useH ? heur(sb, dists) : 0;
  if (h0 === Infinity) return null;
  const heap = [[h0, 0, 0, 0]];
  let seq = 1;
  const less = (a, b) => a[0] < b[0] || (a[0] === b[0] && (useH ? (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])) : a[2] < b[2]));
  const hpush = e => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const hpop = () => {
    const top = heap[0], last = heap.pop();
    if (heap.length) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };
  let expanded = 0;
  while (heap.length) {
    const [, g, , s] = hpop();
    if (g !== gs[s]) continue; // stale entry: the state was re-opened cheaper
    const bs = bxs[s];
    if (solved(bs)) {
      // Reconstruction walks the player between pushes; a throw here (walk
      // unreachable) must fail this search, never crash the whole run.
      try {
        const path = [];
        for (let t = s; t > 0; t = par[t]) path.push(t);
        path.reverse();
        let moves = '', player = sp, cur = sb.slice();
        for (const t of path) {
          const f = frm[t], tt = tto[t], d = tt - f;
          for (const b of cur) occ[b] = 1;
          moves += walkTo(w, wall, occ, player, f - d);
          for (const b of cur) occ[b] = 0;
          moves += DIRS.find(([dx, dy]) => dy * w + dx === d)[3];
          player = f;
          cur = cur.map(b => (b === f ? tt : b));
        }
        return {pushes: path.length, moves};
      } catch { return null; }
    }
    if (expanded >= budget) return null;
    expanded++;
    if (++TOT > TOTMAX) return null;
    for (const b of bs) occ[b] = 1;
    const here = ++stamp;
    regionLow(w, wall, occ, ply[s], seen, here, stack);
    for (let bi = 0; bi < bs.length; bi++) {
      const b = bs[bi];
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx, t = b + d;
        if (seen[b - d] !== here || wall[t] || occ[t] || !live[t]) continue;
        occ[b] = 0; occ[t] = 1;
        let key = null, next = null;
        if (!frozen(w, wall, occ, goal, t)) {
          next = bs.slice(); next[bi] = t; next.sort((x, y) => x - y);
          key = keyOf(next, regionLow(w, wall, occ, b, mark, ++stamp, stack));
        }
        occ[t] = 0; occ[b] = 1;
        if (key === null) continue;
        const h = useH ? heur(next, dists) : 0;
        if (h === Infinity) continue;
        const old = idx.get(key);
        if (old !== undefined) {
          if (g + 1 >= gs[old]) continue;
          gs[old] = g + 1; par[old] = s; frm[old] = b; tto[old] = t;
          hpush([g + 1 + h, g + 1, seq++, old]);
          continue;
        }
        const ni = bxs.length;
        idx.set(key, ni);
        bxs.push(next); ply.push(b); par.push(s); frm.push(b); tto.push(t); gs.push(g + 1);
        hpush([g + 1 + h, g + 1, seq++, ni]);
      }
    }
    for (const b of bs) occ[b] = 0;
  }
  return null;
}

// The level's rooms: cut the dead squares of the player's (hallway) row out of
// the floor; each part left holding boxes is a room. Null when the level is no
// set of rooms (then the whole level is searched as one).
function roomsOf(level) {
  const {w, wall, goals, boxes, player} = level;
  const live = liveSq(w, wall, goals);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some(b => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i] + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) { part[q] = rooms.length; squares.push(q); }
      }
    }
    rooms.push({squares, boxes: [], goals: []});
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]].goals.push(g);
  if (goals.some(g => part[g] < 0) || rooms.some(r => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter(r => r.boxes.length > 0);
}

function endOf(w, from, moves) {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

function solveLevel(text) {
  const level = parse(text);
  // Malformed levels (no player, boxes != goals, a goal on a wall square —
  // incl. the border, which parse walls over) hang or crash the searches
  // below; they are unsolvable by construction, so answer '' up front.
  if (level.player < 0 || level.boxes.length !== level.goals.length || level.goals.some(g => level.wall[g])) return '';
  const rooms = roomsOf(level);
  if (rooms === null) {
    const r = search(level, true, WB) ?? search(level, false, WB);
    return r ? r.moves : '';
  }
  let player = level.player, moves = '';
  const done = new Set();
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const o of rooms) {
      if (o === room) continue;
      for (const b of done.has(o) ? o.goals : o.boxes) wall[b] = 1;
    }
    const sub = {w: level.w, h: level.h, wall, goals: room.goals.slice().sort((a, b) => a - b), boxes: room.boxes.slice().sort((a, b) => a - b), player};
    const r = search(sub, true, AB) ?? search(sub, false, BB);
    if (!r) {
      // The room pass has no backtracking: a room solved first can seal the
      // only way into a later room (e.g. its goal sits in its only doorway),
      // so a room failure does not prove the level unsolvable. Fall back to
      // the whole-level search — the same path used when roomsOf is null.
      const whole = search(level, true, WB) ?? search(level, false, WB);
      return whole ? whole.moves : '';
    }
    moves += r.moves;
    player = endOf(level.w, player, r.moves);
    done.add(room);
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), {recursive: true});
  writeFileSync(join(O, id, 'solution.txt'), moves + '\n');
}
