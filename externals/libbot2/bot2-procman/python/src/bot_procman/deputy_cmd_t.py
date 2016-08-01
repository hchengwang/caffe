"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

import cStringIO as StringIO
import struct

class deputy_cmd_t(object):
    __slots__ = ["name", "nickname", "group", "pid", "actual_runid", "exit_code", "cpu_usage", "mem_vsize_bytes", "mem_rss_bytes", "sheriff_id", "auto_respawn"]

    def __init__(self):
        self.name = ""
        self.nickname = ""
        self.group = ""
        self.pid = 0
        self.actual_runid = 0
        self.exit_code = 0
        self.cpu_usage = 0.0
        self.mem_vsize_bytes = 0
        self.mem_rss_bytes = 0
        self.sheriff_id = 0
        self.auto_respawn = False

    def encode(self):
        buf = StringIO.StringIO()
        buf.write(deputy_cmd_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        __name_encoded = self.name.encode('utf-8')
        buf.write(struct.pack('>I', len(__name_encoded)+1))
        buf.write(__name_encoded)
        buf.write("\0")
        __nickname_encoded = self.nickname.encode('utf-8')
        buf.write(struct.pack('>I', len(__nickname_encoded)+1))
        buf.write(__nickname_encoded)
        buf.write("\0")
        __group_encoded = self.group.encode('utf-8')
        buf.write(struct.pack('>I', len(__group_encoded)+1))
        buf.write(__group_encoded)
        buf.write("\0")
        buf.write(struct.pack(">iiifqqib", self.pid, self.actual_runid, self.exit_code, self.cpu_usage, self.mem_vsize_bytes, self.mem_rss_bytes, self.sheriff_id, self.auto_respawn))

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = StringIO.StringIO(data)
        if buf.read(8) != deputy_cmd_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return deputy_cmd_t._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = deputy_cmd_t()
        __name_len = struct.unpack('>I', buf.read(4))[0]
        self.name = buf.read(__name_len)[:-1].decode('utf-8', 'replace')
        __nickname_len = struct.unpack('>I', buf.read(4))[0]
        self.nickname = buf.read(__nickname_len)[:-1].decode('utf-8', 'replace')
        __group_len = struct.unpack('>I', buf.read(4))[0]
        self.group = buf.read(__group_len)[:-1].decode('utf-8', 'replace')
        self.pid, self.actual_runid, self.exit_code, self.cpu_usage, self.mem_vsize_bytes, self.mem_rss_bytes, self.sheriff_id, self.auto_respawn = struct.unpack(">iiifqqib", buf.read(37))
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if deputy_cmd_t in parents: return 0
        tmphash = (0xed5fbe5982ac8353) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff)  + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if deputy_cmd_t._packed_fingerprint is None:
            deputy_cmd_t._packed_fingerprint = struct.pack(">Q", deputy_cmd_t._get_hash_recursive([]))
        return deputy_cmd_t._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

