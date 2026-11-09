"""Length-prefixed messages between the local policy and Isaac processes."""

import pickle
import struct


def _read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.recv(remaining)
        if not chunk:
            raise EOFError("The other process closed the policy connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send(stream, value):
    payload = pickle.dumps(value, protocol=5)
    stream.sendall(struct.pack("!I", len(payload)) + payload)


def receive(stream):
    size = struct.unpack("!I", _read_exact(stream, 4))[0]
    return pickle.loads(_read_exact(stream, size))


class RemoteController:
    def __init__(self, stream):
        self.stream = stream

    def reset(self, seed):
        send(self.stream, {"kind": "reset", "seed": seed})
        response = receive(self.stream)
        if response.get("error"):
            raise RuntimeError(response["error"])

    def act(self, images, state):
        send(self.stream, {"kind": "action", "images": images, "state": state})
        response = receive(self.stream)
        if response.get("error"):
            raise ValueError(response["error"])
        return response["action"]
