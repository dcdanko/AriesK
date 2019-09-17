
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

    def search(self, kmer, outer_radius, inner_radius):
        self.socket.send_string(f'{kmer} {outer_radius} {inner_radius}')
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
        if auto_start:
            self.main_loop()

    def main_loop(self):
        self.running = True
        while self.running:
            msg = self.socket.recv_string()
            if self.logger:
                logger(f'MESSAGE_RECEIVED: {msg}')
            if msg == SHUTDOWN_MSG:
                break
            kmer, outer_radius, inner_radius = msg.split()
            results = self.grid.search(kmer, float(outer_radius), inner_radius=float(inner_radius))
            results = '\n'.join(list(results))
            self.socket.send_string(results)
        self.running = False

    @classmethod
    def from_filepath(cls, port, filepath, auto_start=False):
        grid = GridCoverSearcher.from_filepath(filepath)
        return cls(port, grid, auto_start=auto_start)
