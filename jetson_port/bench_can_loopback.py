#!/usr/bin/env python3
"""SAFETY BENCH TEST: verify the DIY F446 panda over serial with CAN in internal LOOPBACK mode.

Loopback mode (CAN_BTR_SILM|LBKM) loops TX->RX *inside* the STM32 bxCAN peripheral — NOTHING goes
on the physical bus, transceivers optional. This proves the panda protocol + CAN peripheral +
firmware work with ZERO risk. RUN THIS (and pass it) before ever connecting to the car.

Steps: connect serial -> set safety SILENT (outputs blocked) -> enable loopback -> send a CAN
frame on bus 0 -> read it back -> verify address+data match.

Usage: python3 bench_can_loopback.py [/dev/ttyACM0]
"""
import sys, struct, time, serial

SER_SYNC=0x5A; SER_HACK=0x79; SER_NACK=0x1F; SER_CK=0xAB; SER_HDR=7
DLC_TO_LEN=[0,1,2,3,4,5,6,7,8,12,16,20,24,32,48,64]; LEN_TO_DLC={l:d for d,l in enumerate(DLC_TO_LEN)}

def cksum(b): 
    r=0
    for x in b: r^=x
    return r

class SerialPanda:
    def __init__(self, dev):
        self.s=serial.Serial(dev, 1500000, timeout=0.2)
        self.s.reset_input_buffer(); self.s.reset_output_buffer()
    def _txn(self, endpoint, tx=b"", max_rx=0):
        hdr=bytearray(SER_HDR)
        hdr[0]=SER_SYNC; hdr[1]=endpoint
        hdr[2]=len(tx)&0xFF; hdr[3]=len(tx)>>8
        hdr[4]=max_rx&0xFF; hdr[5]=max_rx>>8
        hdr[6]=SER_CK ^ cksum(hdr[:6])           # device folds to 0
        self.s.write(hdr)
        ack=self.s.read(1)
        if ack!=bytes([SER_HACK]): return None
        if tx:
            self.s.write(tx); self.s.write(bytes([SER_CK ^ cksum(tx)]))
        rh=self.s.read(3)
        if len(rh)<3 or rh[0]!=SER_HACK: return None
        rlen=rh[1]|(rh[2]<<8)
        data=self.s.read(rlen) if rlen else b""
        self.s.read(1)                           # trailing checksum
        return data
    # control transfer: [request:1][param1:2][param2:2][length:2] = ControlPacket_t (packed)
    def control_write(self, req, p1, p2):
        pkt=struct.pack("<BHHH", req, p1, p2, 0)
        return self._txn(0, pkt, 0)
    def control_read(self, req, p1, p2, length):
        pkt=struct.pack("<BHHH", req, p1, p2, length)
        return self._txn(0, pkt, length)
    def can_send(self, addr, dat, bus):
        ext=1 if addr>=0x800 else 0; dlc=LEN_TO_DLC[len(dat)]
        h=bytearray(6); w=(addr<<3)|(ext<<2)
        h[0]=(dlc<<4)|(bus<<1); h[1]=w&0xFF; h[2]=(w>>8)&0xFF; h[3]=(w>>16)&0xFF; h[4]=(w>>24)&0xFF
        h[5]=cksum(h[:5]+dat)
        self._txn(3, bytes(h)+dat, 0)
    def can_recv(self):
        # Parse the packed CAN stream with validation + resync.
        # Without this, a single byte of misalignment corrupts every following
        # frame: the bus/addr/len fields are read at wrong offsets, producing
        # impossible IDs (>0x7FF), impossible lengths (>8) and bus-0 payloads
        # mislabelled as bus 1. On a bad header we drop ONE byte and retry,
        # which re-locks onto the next real frame boundary.
        raw = self._txn(1, b"", 2040)
        out = []
        d = raw or b""
        i = 0
        n = len(d)
        while i + 6 <= n:
            dlc = d[i] >> 4
            bus = (d[i] >> 1) & 0x7
            dl = DLC_TO_LEN[dlc]
            w = d[i+1] | (d[i+2] << 8) | (d[i+3] << 16) | (d[i+4] << 24)
            addr = w >> 3
            ext = (w >> 2) & 1
            # validity checks: standard IDs are 11-bit, classic CAN is <=8 bytes,
            # and this firmware only has buses 0-2.
            ok = (dl <= 8) and (bus <= 2) and (i + 6 + dl <= n) and \
                 (addr <= 0x1FFFFFFF if ext else addr <= 0x7FF)
            if not ok:
                i += 1          # resync: slide one byte and retry
                continue
            out.append((addr, bytes(d[i+6:i+6+dl]), bus))
            i += 6 + dl
        return out

def main():
    dev=sys.argv[1] if len(sys.argv)>1 else "/dev/ttyACM0"
    print(f"=== F446 PANDA CAN LOOPBACK BENCH TEST on {dev} ===")
    p=SerialPanda(dev)
    # health check first
    h=p.control_read(0xd2, 0, 0, 64)             # 0xd2 = get health
    print(f"health resp: {len(h) if h else 0} bytes {'(panda responding)' if h else '(NO RESPONSE — check link)'}")
    print("set safety SILENT (0xdc, mode=0) — outputs blocked, safe")
    p.control_write(0xdc, 0, 0)
    print("enable CAN loopback (0xe5, 1) — TX loops to RX internally, nothing on the wire")
    p.control_write(0xe5, 1, 0)
    time.sleep(0.1)
    # send a test frame + read back
    test_addr=0x123; test_data=bytes([0xDE,0xAD,0xBE,0xEF])
    print(f"send CAN: addr=0x{test_addr:x} data={test_data.hex()} bus=0")
    p.can_send(test_addr, test_data, 0)
    time.sleep(0.05)
    got=p.can_recv()
    print(f"received {len(got)} frame(s): {[(hex(a),d.hex(),b) for a,d,b in got]}")
    ok=any(a==test_addr and d==test_data for a,d,b in got)
    print("\n"+("✅ PASS — loopback frame matches. Panda protocol + CAN peripheral work."
                 if ok else "❌ FAIL — frame not looped back. Debug before proceeding."))
    print("(Loopback = zero bus activity. Passing this is REQUIRED before any car connection.)")

if __name__=="__main__": main()
