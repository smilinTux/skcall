#!/usr/bin/env python3
"""Minimal TURN Allocate probe (RFC 5766/8656) over udp, tcp or tls. Stdlib only.

Usage: turn_probe.py HOST PORT {udp,tcp,tls} [--sni NAME] [--cred-file F] [--hold SECONDS] [--refresh-every SECONDS]
cred-file: two lines, username and password (TURN REST ephemeral creds).
Without creds it only checks that a TURN server answers (expects 401 + realm/nonce).
With creds it allocates, prints the relayed address, then frees it (Refresh lifetime 0).
--hold keeps the allocation open that long, sending a Refresh every --refresh-every s (default 60).
Never prints credentials.
"""
import argparse, hashlib, hmac, os, socket, ssl, struct, sys, time, zlib

MAGIC = 0x2112A442


def attr(t, v):
    pad = (4 - len(v) % 4) % 4
    return struct.pack("!HH", t, len(v)) + v + b"\0" * pad


def msg(mtype, tid, attrs, key=None):
    body = b"".join(attrs)
    if key is not None:
        hdr = struct.pack("!HHI", mtype, len(body) + 24, MAGIC) + tid
        body += attr(0x0008, hmac.new(key, hdr + body, hashlib.sha1).digest())
    hdr = struct.pack("!HHI", mtype, len(body) + 8, MAGIC) + tid
    crc = (zlib.crc32(hdr + body) ^ 0x5354554E) & 0xFFFFFFFF
    body += attr(0x8028, struct.pack("!I", crc))
    return struct.pack("!HHI", mtype, len(body), MAGIC) + tid + body


def parse(data):
    mtype, mlen, _ = struct.unpack("!HHI", data[:8])
    attrs, i = {}, 20
    while i < 20 + mlen:
        t, l = struct.unpack("!HH", data[i:i + 4])
        attrs.setdefault(t, data[i + 4:i + 4 + l])
        i += 4 + l + ((4 - l % 4) % 4)
    return mtype, attrs


def xaddr(v):
    port = struct.unpack("!H", v[2:4])[0] ^ (MAGIC >> 16)
    ip = struct.unpack("!I", v[4:8])[0] ^ MAGIC
    return f"{socket.inet_ntoa(struct.pack('!I', ip))}:{port}"


def errcode(at):
    e = at.get(0x0009, b"\0\0\0\0")
    return (e[2] & 7) * 100 + e[3], e[4:].decode(errors="replace")


class Conn:
    def __init__(self, host, port, mode, sni):
        self.mode, self.buf = mode, b""
        if mode == "udp":
            self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.s.settimeout(5)
            self.s.connect((host, port))
            return
        raw = socket.create_connection((host, port), timeout=8)
        if mode == "tls":
            self.s = ssl.create_default_context().wrap_socket(raw, server_hostname=sni or host)
            cn = dict(x[0] for x in self.s.getpeercert()["subject"]).get("commonName")
            print(f"  TLS ok: {self.s.version()} cert CN={cn}")
        else:
            self.s = raw
        self.s.settimeout(8)

    def rt(self, data):
        if self.mode == "udp":
            self.s.send(data)
            return self.s.recv(4096)
        self.s.sendall(data)
        while True:
            if len(self.buf) >= 20:
                n = struct.unpack("!H", self.buf[2:4])[0] + 20
                if len(self.buf) >= n:
                    out, self.buf = self.buf[:n], self.buf[n:]
                    return out
            chunk = self.s.recv(4096)
            if not chunk:
                raise ConnectionError("closed by server")
            self.buf += chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("host")
    ap.add_argument("port", type=int)
    ap.add_argument("mode", choices=["udp", "tcp", "tls"])
    ap.add_argument("--sni")
    ap.add_argument("--cred-file")
    ap.add_argument("--hold", type=float, default=0)
    ap.add_argument("--refresh-every", type=float, default=60)
    a = ap.parse_args()
    print(f"probe {a.mode}://{a.host}:{a.port}" + (f" sni={a.sni}" if a.sni else ""))
    transport = attr(0x0019, struct.pack("!B3x", 17))  # REQUESTED-TRANSPORT udp
    try:
        c = Conn(a.host, a.port, a.mode, a.sni)
        mtype, at = parse(c.rt(msg(0x0003, os.urandom(12), [transport])))
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return 2
    if mtype != 0x0113 or 0x0014 not in at:
        print(f"  FAIL: unexpected response type 0x{mtype:04x}")
        return 2
    realm, nonce = at[0x0014], at[0x0015]
    print(f"  TURN answered: error {errcode(at)[0]} with realm={realm.decode()}")
    if not a.cred_file:
        return 0
    user, pwd = [l.strip() for l in open(a.cred_file).read().splitlines()[:2]]
    key = hashlib.md5(f"{user}:{realm.decode()}:{pwd}".encode()).digest()
    auth = [attr(0x0006, user.encode()), attr(0x0014, realm), attr(0x0015, nonce)]
    mtype, at = parse(c.rt(msg(0x0003, os.urandom(12), [transport] + auth, key)))
    if mtype != 0x0103:
        print("  FAIL: Allocate error %d %s" % errcode(at))
        return 3
    print(f"  ALLOCATE OK: relayed address {xaddr(at[0x0016])}, lifetime {struct.unpack('!I', at[0x000D])[0]}s")
    t0 = time.monotonic()
    while time.monotonic() - t0 < a.hold:
        time.sleep(min(a.refresh_every, a.hold - (time.monotonic() - t0)))
        mtype, at = parse(c.rt(msg(0x0004, os.urandom(12), [attr(0x000D, struct.pack("!I", 600))] + auth, key)))
        if mtype == 0x0114 and errcode(at)[0] == 438 and 0x0015 in at:
            # Stale Nonce: retry once with the fresh nonce, like a real TURN client
            nonce = at[0x0015]
            auth = [attr(0x0006, user.encode()), attr(0x0014, realm), attr(0x0015, nonce)]
            print("  (438 stale nonce, retrying with the new nonce)")
            mtype, at = parse(c.rt(msg(0x0004, os.urandom(12), [attr(0x000D, struct.pack("!I", 600))] + auth, key)))
        ok = mtype == 0x0104
        print(f"  t={time.monotonic() - t0:.0f}s refresh {'ok' if ok else 'FAILED %d %s' % errcode(at)}", flush=True)
        if not ok:
            return 4
    mtype, _ = parse(c.rt(msg(0x0004, os.urandom(12), [attr(0x000D, struct.pack("!I", 0))] + auth, key)))
    print("  freed allocation" if mtype == 0x0104 else f"  refresh(0) answered 0x{mtype:04x}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
