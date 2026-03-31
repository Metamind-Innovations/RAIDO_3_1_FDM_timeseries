import io
import pickle
import socket
import struct
from typing import Any, Dict

import torch


def recvall(sock: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Socket connection closed unexpectedly.")
        data += packet
    return data


def send_msg(sock: socket.socket, obj: Any) -> None:
    payload = pickle.dumps(obj)
    header = struct.pack("!Q", len(payload))
    sock.sendall(header + payload)


def recv_msg(sock: socket.socket) -> Any:
    raw_len = recvall(sock, 8)
    msg_len = struct.unpack("!Q", raw_len)[0]
    payload = recvall(sock, msg_len)
    return pickle.loads(payload)


def state_dict_to_bytes(state_dict: Dict[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO()
    cpu_state = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in state_dict.items()}
    torch.save(cpu_state, buffer)
    return buffer.getvalue()


def bytes_to_state_dict(blob: bytes) -> Dict[str, torch.Tensor]:
    buffer = io.BytesIO(blob)
    return torch.load(buffer, map_location="cpu")
