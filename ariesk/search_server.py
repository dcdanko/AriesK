
import zmq
from json import dumps, loads
from time import time

from ariesk.searcher import GridCoverSearcher

'''
Allowed client->server message terms. *mandatory
    type: search|shutdown
    query_type: sequence|file|multiseq
    query: <string>
    outer_radius: <float>
    inner_radius: <float>
    inner_metric: hamming|needle|none
    search_mode: coarse|full
'''


class SearchClient:

    def __init__(self, port, callback=None):
        self.callback = callback
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f'tcp://127.0.0.1:{port}')

    def search(self, query, outer_radius, inner_radius, **kwargs):
        msg = {
            'type': 'search',
            'query_type': 'sequence',
            'query': query,
            'outer_radius': outer_radius,
            'inner_radius': inner_radius,
            'search_mode': 'full',
            'inner_metric': 'needle',
        }
        msg.update(kwargs)
        self.socket.send_string(dumps(msg))
        results = self.socket.recv_string().split('\n')
        return results

    def handshake(self):
        self.socket.send_string(dumps({'type': 'handshake'}))
        self.socket.recv_string()

    def send_shutdown(self):
        self.socket.send_string(dumps({'type': 'shutdown'}))


class SearchServer:

    def __init__(self, port, grid_cover, auto_start=False, logger=None):
        self.grid = grid_cover
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f'tcp://*:{port}')
        self.running = False
        self.logger = logger
        if self.logger:
            self.grid.add_logger(logger)
        if auto_start:
            self.main_loop()

    def main_loop(self):
        self.running = True
        while self.running:
            msg = loads(self.socket.recv_string())
            if self.logger:
                self.logger(f'MESSAGE_RECEIVED: {msg}')
            start = time()
            if msg['type'] == 'shutdown':
                break
            elif msg['type'] == 'handshake':
                results = dumps({'type': 'handshake'})
            elif msg['query_type'] == 'file':
                results = self.full_file_search(msg)
            elif msg['search_mode'] == 'full':
                results = self.full_search(msg)
            elif msg['search_mode'] == 'coarse':
                results = self.coarse_search(msg)
            elapsed = time() - start
            self.socket.send_string(results)
            if self.logger:
                self.logger(f'TIME_TO_REPLY: {elapsed:.5}s')
                # self.logger(f'MESSAGE_SENT: {results}')
        self.running = False

    def full_search(self, msg):
        results = self.grid.py_search(
            msg['query'],
            msg['outer_radius'],
            inner_radius=msg['inner_radius'],
            inner_metric=msg['inner_metric'],
        )
        results = '\n'.join(list(results))
        return results

    def full_file_search(self, msg):
        self.grid.file_search(
            msg['query'],
            msg['result_file'],
            msg['outer_radius'],
            inner_radius=msg['inner_radius'],
            inner_metric=msg['inner_metric'],
        )
        results = 'DONE'
        return results

    def coarse_search(self, msg):
        results = self.grid.py_coarse_search(
            msg['query'],
            msg['outer_radius'],
        )
        results = '\n'.join([str(el) for el in results])
        return results

    @classmethod
    def from_filepath(cls, port, filepath, **kwargs):
        grid = GridCoverSearcher.from_filepath(filepath)
        return cls(port, grid, **kwargs)
