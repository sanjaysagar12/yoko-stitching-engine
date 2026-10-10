#!/usr/bin/env python3
"""Minimal Seamly2D evaluator: .sm2d + measurements -> piece outlines in real cm.

Tool maths follows the Seamly2D source (FashionFreedom/Seamly2D, develop):
  endLine           Qt QLineF(base, base+(100,0)); setAngle(a); setLength(L)   (y-down screen, ccw angle)
  alongLine         point at distance L from firstPoint toward secondPoint
  normal            QLineF(first,second).normalVector() = (dy,-dx); angle += extra; setLength(L)
  intersectXY       (first.x, second.y)
  height            foot of perpendicular from basePoint onto line p1Line-p2Line
  lineIntersectAxis axis (base, angle) x infinite line p1Line-p2Line
  curveIntersectAxis nearest crossing of ray (base, angle) with the curve (else the opposite ray)
  shoulder          circle(pShoulder, length) x ray p1Line->p2Line, beyond p2Line (else p2Line)
  cutSpline         point at arc length L from spline start (L clamped to [1 mm, full-1 mm])
  simpleInteractive spline: P2 = P1 + len1@angle1, P3 = P4 + len2@angle2
  pathInteractive   segment i: P_i, P_i+len2_i@angle2_i, P_i+1+len1_i+1@angle1_i+1, P_i+1
  multisize         value = base + (size-baseSize)/2*size_increase + (height-baseHeight)/6*height_increase
Anything else raises NotImplementedError (never silently wrong).
"""
import math
import re
import sys
import xml.etree.ElementTree as ET


# ----------------------------------------------------------------------------- measurements
def load_smms(path, size, height):
    root = ET.parse(path).getroot()
    base_size = float(root.find("size").get("base"))
    base_height = float(root.find("height").get("base"))
    k_size = (size - base_size) / 2.0
    k_height = (height - base_height) / 6.0
    out = {}
    for m in root.find("body-measurements"):
        out[m.get("name")] = (float(m.get("base"))
                              + k_size * float(m.get("size_increase", 0))
                              + k_height * float(m.get("height_increase", 0)))
    return out


def load_smis(path):
    root = ET.parse(path).getroot()
    return {m.get("name"): float(m.get("value").strip()) for m in root.find("body-measurements")}


# ----------------------------------------------------------------------------- formulas
def _ternary(s):
    """muParser 'c ? a : b' -> python conditional; also converts ternaries nested inside parentheses."""
    out, i = [], 0
    while i < len(s):
        if s[i] == "(":
            depth, j = 1, i + 1
            while j < len(s) and depth:
                depth += (s[j] == "(") - (s[j] == ")")
                j += 1
            out.append("(" + _ternary(s[i + 1:j - 1]) + ")")
            i = j
        else:
            out.append(s[i])
            i += 1
    return _ternary_top("".join(out))


def _ternary_top(s):
    depth = 0
    q = -1
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "?" and depth == 0:
            q = i
            break
    if q < 0:
        return s
    cond, rest = s[:q], s[q + 1:]
    depth = level = 0
    c = -1
    for i, ch in enumerate(rest):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and ch == "?":
            level += 1
        elif depth == 0 and ch == ":":
            if level == 0:
                c = i
                break
            level -= 1
    a, b = rest[:c], rest[c + 1:]
    return f"(({_ternary_top(a)}) if ({_ternary_top(cond)}) else ({_ternary_top(b)}))"


_D = math.radians
FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sinD": lambda x: math.sin(_D(x)), "cosD": lambda x: math.cos(_D(x)), "tanD": lambda x: math.tan(_D(x)),
    "asinD": lambda x: math.degrees(math.asin(x)), "acosD": lambda x: math.degrees(math.acos(x)),
    "atanD": lambda x: math.degrees(math.atan(x)),
    "sqrt": math.sqrt, "abs": abs, "min": min, "max": max, "exp": math.exp, "log": math.log,
    "ln": math.log, "log10": math.log10, "log2": math.log2, "rint": round, "sign": lambda x: (x > 0) - (x < 0),
}


def qt_angle(p, q):
    """QLineF(p,q).angle(): degrees ccw, y axis pointing up on screen."""
    return math.degrees(math.atan2(-(q[1] - p[1]), q[0] - p[0])) % 360.0


def move(p, length, angle):
    a = math.radians(angle)
    return (p[0] + length * math.cos(a), p[1] - length * math.sin(a))


def bez(p0, p1, p2, p3, n):
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append((u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                    u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]))
    return out


def poly_length(P):
    return sum(math.dist(P[i], P[i + 1]) for i in range(len(P) - 1))


def ray_polyline(base, angle, P):
    """All intersections of ray(base,angle) with polyline P, as points."""
    d = (math.cos(math.radians(angle)), -math.sin(math.radians(angle)))
    hits = []
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        e = (b[0] - a[0], b[1] - a[1])
        den = d[0] * e[1] - d[1] * e[0]
        if abs(den) < 1e-14:
            continue
        w = (a[0] - base[0], a[1] - base[1])
        t = (w[0] * e[1] - w[1] * e[0]) / den        # along ray
        u = (w[0] * d[1] - w[1] * d[0]) / den        # along segment
        if t >= -1e-12 and -1e-12 <= u <= 1 + 1e-12:
            hits.append((base[0] + t * d[0], base[1] + t * d[1]))
    return hits


class Pattern:
    def __init__(self, sm2d_path, meas):
        self.root = ET.parse(sm2d_path).getroot()
        self.meas = {k: float(v) for k, v in meas.items()}
        self.elems = {}            # id -> element (calculation objects)
        self.names = {}            # point name -> id
        for el in self.root.iter():
            if el.tag in ("point", "line", "spline", "arc", "operation") and el.get("id"):
                if el.get("type") in ("modeling", "modelingSpline", "modelingPath"):
                    continue
                self.elems[el.get("id")] = el
                if el.tag == "point" and el.get("name"):
                    self.names[el.get("name")] = el.get("id")
        self.modeling = {}
        for el in self.root.iter():
            if el.get("type") in ("modeling", "modelingSpline", "modelingPath"):
                self.modeling[el.get("id")] = el.get("idObject")
        self.incr = {i.get("name"): i.get("formula") for i in self.root.iter("increment")}
        self._pt, self._incr_val, self._curve_cache, self._cur = {}, {}, {}, None
        self._curve_by_ends = None

    # -- variable resolution ------------------------------------------------
    def _curve_index(self):
        if self._curve_by_ends is None:
            self._curve_by_ends = {}
            for cid, el in self.elems.items():
                if el.tag != "spline":
                    continue
                t = el.get("type")
                if t in ("cubicBezier", "simpleInteractive"):
                    a, b = el.get("point1"), el.get("point4")
                elif t in ("cubicBezierPath", "pathInteractive"):
                    pp = el.findall("pathPoint")
                    a, b = pp[0].get("pSpline"), pp[-1].get("pSpline")
                else:
                    continue
                na, nb = self.elems[a].get("name"), self.elems[b].get("name")
                kind = "SplPath_" if t in ("cubicBezierPath", "pathInteractive") else "Spl_"
                self._curve_by_ends[f"{kind}{na}_{nb}"] = cid
        return self._curve_by_ends

    def _subcurve_index(self):
        """Seamly names the two halves of a curve split by a point on it (curveIntersectAxis / cutSpline):
        <Spl|SplPath>_<curveStart>_<point> and <Spl|SplPath>_<point>_<curveEnd>."""
        if getattr(self, "_sub", None) is None:
            self._sub = {}
            for pid, el in self.elems.items():
                if el.tag != "point" or el.get("type") not in ("curveIntersectAxis", "cutSpline"):
                    continue
                cid = self._curve_of(el.get("curve") or el.get("spline"))
                c = self.elems.get(cid)
                if c is None or c.tag != "spline":
                    continue
                if c.get("type") in ("cubicBezier", "simpleInteractive"):
                    a, b, kind = c.get("point1"), c.get("point4"), "Spl_"
                else:
                    pp = c.findall("pathPoint")
                    a, b, kind = pp[0].get("pSpline"), pp[-1].get("pSpline"), "SplPath_"
                S, E, X = (self.elems[a].get("name"), self.elems[b].get("name"), el.get("name"))
                self._sub[f"{kind}{S}_{X}"] = (cid, pid, "start")
                self._sub[f"{kind}{X}_{E}"] = (cid, pid, "end")
        return self._sub

    def _ident(self, name):
        if name == "CurrentLength":
            return self._cur
        if name in self.meas:
            return self.meas[name]
        if name.startswith("#") and name in self.incr:
            if name not in self._incr_val:
                self._incr_val[name] = self.ev(self.incr[name])
            return self._incr_val[name]
        idx = self._curve_index()
        if name in idx:
            return self.curve_length(idx[name])
        sub = self._subcurve_index()
        if name in sub:
            cid, pid, part = sub[name]
            P = self.curve_points(cid, 500)
            s, _ = self._project(P, self.point(pid))
            return s if part == "start" else poly_length(P) - s
        for pre, fn in (("AngleLine_", lambda a, b: qt_angle(a, b)),
                        ("Line_", lambda a, b: math.dist(a, b))):
            if name.startswith(pre):
                parts = name[len(pre):].split("_")
                for i in range(1, len(parts)):
                    a, b = "_".join(parts[:i]), "_".join(parts[i:])
                    if a in self.names and b in self.names:
                        return fn(self.point(self.names[a]), self.point(self.names[b]))
        raise KeyError(f"unresolved variable {name}")

    def ev(self, formula):
        f = formula.strip()
        f = re.sub(r"[A-Za-z_#][A-Za-z0-9_#]*",
                   lambda m: m.group(0) if m.group(0) in FUNCS else repr(float(self._ident(m.group(0)))), f)
        f = f.replace("&&", " and ").replace("||", " or ").replace("^", "**")
        return float(eval(_ternary(f), {"__builtins__": {}}, dict(FUNCS)))  # noqa: S307 (numbers/functions only)

    # -- points -------------------------------------------------------------
    def point(self, pid):
        if pid in self._pt:
            return self._pt[pid]
        el = self.elems[pid]
        t = el.get("type")
        g = el.get
        if t == "single":
            p = (float(g("x")), float(g("y")))
        elif t == "endLine":
            b = self.point(g("basePoint"))
            ang = self.ev(g("angle"))
            p = move(b, self.ev(g("length")), ang)
        elif t == "alongLine":
            a, b = self.point(g("firstPoint")), self.point(g("secondPoint"))
            d = math.dist(a, b)
            self._cur = d
            L = self.ev(g("length"))
            p = (a[0] + (b[0] - a[0]) / d * L, a[1] + (b[1] - a[1]) / d * L)
        elif t == "normal":
            a, b = self.point(g("firstPoint")), self.point(g("secondPoint"))
            self._cur = math.dist(a, b)
            L, extra = self.ev(g("length")), self.ev(g("angle") or "0")
            n_ang = qt_angle((0, 0), ((b[1] - a[1]), -(b[0] - a[0]))) + extra   # normalVector = (dy,-dx)
            p = move(a, L, n_ang)
        elif t == "intersectXY":
            a, b = self.point(g("firstPoint")), self.point(g("secondPoint"))
            p = (a[0], b[1])
        elif t == "height":
            base, a, b = self.point(g("basePoint")), self.point(g("p1Line")), self.point(g("p2Line"))
            dx, dy = b[0] - a[0], b[1] - a[1]
            k = ((base[0] - a[0]) * dx + (base[1] - a[1]) * dy) / (dx * dx + dy * dy)
            p = (a[0] + k * dx, a[1] + k * dy)
        elif t == "lineIntersectAxis":
            base, a, b = self.point(g("basePoint")), self.point(g("p1Line")), self.point(g("p2Line"))
            ang = self.ev(g("angle"))
            d = (math.cos(math.radians(ang)), -math.sin(math.radians(ang)))
            e = (b[0] - a[0], b[1] - a[1])
            den = d[0] * e[1] - d[1] * e[0]
            if abs(den) < 1e-12:
                raise ValueError(f"lineIntersectAxis {g('name')}: axis parallel to line")
            w = (a[0] - base[0], a[1] - base[1])
            tt = (w[0] * e[1] - w[1] * e[0]) / den
            p = (base[0] + tt * d[0], base[1] + tt * d[1])
        elif t == "curveIntersectAxis":
            base = self.point(g("basePoint"))
            ang = self.ev(g("angle"))
            P = self.curve_points(self._curve_of(g("curve")), 400)
            hits = ray_polyline(base, ang, P) or ray_polyline(base, ang + 180, P)
            if not hits:
                raise ValueError(f"curveIntersectAxis {g('name')}: no intersection")
            p = min(hits, key=lambda h: math.dist(h, base))
        elif t == "shoulder":
            p1, p2, ps = self.point(g("p1Line")), self.point(g("p2Line")), self.point(g("pShoulder"))
            self._cur = None
            L = self.ev(g("length"))
            p = self._shoulder(p1, p2, ps, L)
        elif t == "cutSpline":
            cid = self._curve_of(g("spline"))
            P = self.curve_points(cid, 600)
            full = poly_length(P)
            L = min(max(self.ev(g("length")), 0.1), full - 0.1)       # 1 mm clamp, as in Seamly
            acc = 0.0
            p = P[-1]
            for i in range(len(P) - 1):
                s = math.dist(P[i], P[i + 1])
                if acc + s >= L:
                    k = (L - acc) / s
                    p = (P[i][0] + k * (P[i + 1][0] - P[i][0]), P[i][1] + k * (P[i + 1][1] - P[i][1]))
                    break
                acc += s
        else:
            raise NotImplementedError(f"point tool '{t}' (id {pid}, name {g('name')})")
        self._pt[pid] = p
        return p

    @staticmethod
    def _shoulder(p1, p2, ps, L):
        if L <= 0:
            return p2
        base_len = math.dist(p1, p2)
        base_ang = round(qt_angle(p1, p2))
        d = ((p2[0] - p1[0]) / base_len, (p2[1] - p1[1]) / base_len)
        w = (p1[0] - ps[0], p1[1] - ps[1])
        b = w[0] * d[0] + w[1] * d[1]
        c = w[0] ** 2 + w[1] ** 2 - L * L
        disc = b * b - c
        if disc < 0:
            return p2
        cands = sorted([-b - math.sqrt(disc), -b + math.sqrt(disc)])
        for tt in cands:
            q = (p1[0] + tt * d[0], p1[1] + tt * d[1])
            if tt > base_len and round(qt_angle(p1, q)) == base_ang:
                return q
        return p2

    # -- curves -------------------------------------------------------------
    def _curve_of(self, cid):
        return self.modeling.get(cid, cid)

    def _segments(self, cid):
        """List of 4-point Bezier segments for a spline object."""
        if cid in self._curve_cache:
            return self._curve_cache[cid]
        el = self.elems[cid]
        t = el.get("type")
        if t == "cubicBezier":
            segs = [[self.point(el.get(k)) for k in ("point1", "point2", "point3", "point4")]]
        elif t == "cubicBezierPath":
            P = [self.point(pp.get("pSpline")) for pp in el.findall("pathPoint")]
            segs = [P[i:i + 4] for i in range(0, len(P) - 3, 3)]
        elif t == "simpleInteractive":
            p1, p4 = self.point(el.get("point1")), self.point(el.get("point4"))
            p2 = move(p1, self.ev(el.get("length1")), self.ev(el.get("angle1")))
            p3 = move(p4, self.ev(el.get("length2")), self.ev(el.get("angle2")))
            segs = [[p1, p2, p3, p4]]
        elif t == "pathInteractive":
            pp = el.findall("pathPoint")
            P = [self.point(x.get("pSpline")) for x in pp]
            segs = []
            for i in range(len(pp) - 1):
                p2 = move(P[i], self.ev(pp[i].get("length2")), self.ev(pp[i].get("angle2")))
                p3 = move(P[i + 1], self.ev(pp[i + 1].get("length1")), self.ev(pp[i + 1].get("angle1")))
                segs.append([P[i], p2, p3, P[i + 1]])
        else:
            raise NotImplementedError(f"curve type '{t}' (id {cid})")
        self._curve_cache[cid] = segs
        return segs

    def curve_points(self, cid, n=64):
        pts = []
        for s in self._segments(cid):
            seg = bez(*s, n)
            pts += seg if not pts else seg[1:]
        return pts

    def curve_length(self, cid):
        return poly_length(self.curve_points(cid, 500))

    # -- pieces -------------------------------------------------------------
    @staticmethod
    def _project(P, q):
        """Closest point of polyline P to q -> (arc-length position, distance)."""
        best = (0.0, float("inf"))
        acc = 0.0
        for i in range(len(P) - 1):
            a, b = P[i], P[i + 1]
            e = (b[0] - a[0], b[1] - a[1])
            ll = e[0] ** 2 + e[1] ** 2
            k = 0.0 if ll == 0 else max(0.0, min(1.0, ((q[0] - a[0]) * e[0] + (q[1] - a[1]) * e[1]) / ll))
            c = (a[0] + k * e[0], a[1] + k * e[1])
            dd = math.dist(c, q)
            if dd < best[1]:
                best = (acc + k * math.sqrt(ll), dd)
            acc += math.sqrt(ll)
        return best

    @staticmethod
    def _slice(P, s0, s1):
        """Interior vertices of P strictly between arc positions s0 and s1."""
        out, acc = [], 0.0
        for i in range(len(P)):
            if i > 0:
                acc += math.dist(P[i - 1], P[i])
            if s0 + 1e-9 < acc < s1 - 1e-9:
                out.append(P[i])
        return out

    def piece(self, name, tol=0.05, n=64):
        pe = next(p for p in self.root.iter("piece") if p.get("name") == name)
        items = []
        for nd in pe.find("nodes"):
            if (nd.get("excluded") or "false").lower() in ("1", "true"):
                continue
            cid = self.modeling.get(nd.get("idObject"), nd.get("idObject"))
            if nd.get("type") == "NodePoint":
                items.append(("pt", self.point(cid)))
            elif nd.get("type") in ("NodeSpline", "NodeSplinePath"):
                items.append(("crv", self.curve_points(cid, n), (nd.get("reverse") or "0") in ("1", "true")))
            else:
                raise NotImplementedError(f"piece node type {nd.get('type')}")
        out, notes = [], []
        for i, it in enumerate(items):
            if it[0] == "pt":
                out.append(it[1])
                continue
            P, rflag = it[1], it[2]
            prev = items[i - 1] if i > 0 else items[-1]
            nxt = items[(i + 1) % len(items)]
            best = None
            for rev in (False, True):
                Q = P[::-1] if rev else P
                L = poly_length(Q)
                s0, s1 = 0.0, L
                if prev[0] == "pt":
                    s, dd = self._project(Q, prev[1])
                    if dd < tol:
                        s0 = s
                if nxt[0] == "pt":
                    s, dd = self._project(Q, nxt[1])
                    if dd < tol:
                        s1 = s
                if s0 > s1 + 1e-6:
                    continue
                start = Q[0] if s0 == 0.0 else None
                ref = out[-1] if out else None
                gap = 0.0 if ref is None else (math.dist(ref, Q[0]) if s0 == 0.0 else 0.0)
                cost = gap + (0.0 if rev == rflag else 1e-7)
                if best is None or cost < best[0]:
                    best = (cost, Q, s0, s1, gap)
            if best is None:
                raise ValueError(f"{name}: cannot orient curve node {i}")
            _, Q, s0, s1, gap = best
            if gap > 1e-3:
                notes.append(f"node {i}: gap {gap:.4f} cm to previous point")
            seg = self._slice(Q, s0, s1)
            if s0 == 0.0:
                seg = [Q[0]] + seg
            if s1 == poly_length(Q):
                seg = seg + [Q[-1]]
            out += seg
        clean = []
        for p in out:
            if not clean or math.dist(clean[-1], p) > 1e-9:
                clean.append(p)
        if len(clean) > 1 and math.dist(clean[0], clean[-1]) < 1e-9:
            clean.pop()
        attrs = dict(pe.attrib)
        d = pe.find("data")
        if d is not None:
            attrs.update({"quantity": d.get("quantity"), "onFold": d.get("onFold")})
        return clean, notes, attrs

    def piece_names(self):
        return [p.get("name") for p in self.root.iter("piece")]


def polygon_area(P):
    return 0.5 * sum(P[i][0] * P[(i + 1) % len(P)][1] - P[(i + 1) % len(P)][0] * P[i][1] for i in range(len(P)))


def perimeter(P):
    return sum(math.dist(P[i], P[(i + 1) % len(P)]) for i in range(len(P)))


def self_intersections(P):
    """Count properly crossing non-adjacent edge pairs (O(n^2); fine for a few hundred vertices)."""
    n = len(P)

    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    cnt = 0
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = P[j], P[(j + 1) % n]
            if ccw(a, c, d) * ccw(b, c, d) < 0 and ccw(a, b, c) * ccw(a, b, d) < 0:
                cnt += 1
    return cnt


if __name__ == "__main__":
    sm2d, meas_path = sys.argv[1], sys.argv[2]
    meas = load_smis(meas_path) if meas_path.endswith(".smis") else load_smms(meas_path, float(sys.argv[3]), float(sys.argv[4]))
    pat = Pattern(sm2d, meas)
    for nm in pat.piece_names():
        try:
            poly, notes, _ = pat.piece(nm)
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            print(f"{nm:22s} {len(poly):4d} pts  w={max(xs)-min(xs):7.2f} h={max(ys)-min(ys):7.2f} cm {notes}")
        except Exception as e:  # noqa: BLE001
            print(f"{nm:22s} FAILED: {type(e).__name__}: {e}")
