
import zmq

from ariesk.searcher import GridCoverSearcher

RESULTS_DONE_MSG = 'DONE'
SHUTDOWN_MSG = 'SHUTDOWN'


class SearchClient:

    def __init__(self, port, callback=None):
        self.callback = callback
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f'tcp://127.0.0.1:{port}')

    def search(self, kmer, outer_radius, inner_radius, fast=False):
        fast = 'FAST' if fast else 'SLOW'
        self.socket.send_string(f'FULL {kmer} {outer_radius} {inner_radius} {fast}')
        results = self.socket.recv_string()
        for result in results.split('\n'):
            if self.callback:
                self.callback(result)
            yield result

    def coarse_search(self, kmer, outer_radius):
        self.socket.send_string(f'COARSE {kmer} {outer_radius}')
        results = self.socket.recv_string()
        for result in results.split('\n'):
            if self.callback:
                self.callback(result)
            yield result

    def send_shutdown(self):
        self.socket.send_string(SHUTDOWN_MSG)


class SearchServer:

    def __init__(self, port, grid_cover, auto_start=False, logger=None):
        self.grid = grid_cover
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f'tcp://*:{port}')
        self.running = False
        self.logger = logger
        if auto_start:
            self.main_loop()

    def main_loop(self):
        self.running = True
        while self.running:
            msg = self.socket.recv_string()
            if self.logger:
                self.logger(f'MESSAGE_RECEIVED: {msg}')
            if msg == SHUTDOWN_MSG:
                break
            tkns = msg.split()
            mode, tkns = tkns[0], tkns[1:]
            if mode == 'FULL':
                kmer, outer_radius, inner_radius, fast = tkns
                results = self.grid.search(
                    kmer,
                    float(outer_radius),
                    inner_radius=float(inner_radius),
                    fast_search=(fast == 'FAST')
                )
                results = '\n'.join(list(results))
            elif mode == 'COARSE':
                kmer, outer_radius = tkns
                results = self.grid._coarse_search(
                    kmer,
                    float(outer_radius),
                )
                results = '\n'.join([str(el) for el in results])
            self.socket.send_string(results)
        self.running = False

    @classmethod
    def from_filepath(cls, port, filepath, **kwargs):
        grid = GridCoverSearcher.from_filepath(filepath)
        return cls(port, grid, **kwargs)
