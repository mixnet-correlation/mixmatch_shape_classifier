from pathlib import Path


SEED            = 0
DEVICE          = 'gpu'
EXPERIMENTS     = [1, 2, 5, 6, 7]

ACK_SIZE        = 53
PAIRS           = {'train': 24500,
                   'val':   5250,
                   'test':  5250}

ORIGINS         = ['initiator', 'responder']
DIRECTIONS      = ['from_gateway', 'to_gateway']
PACKET_TYPES    = ['data', 'ack']
CHANNELS        = [0,          1]

ROOTPATH        = Path(__file__).parent.resolve()
DATAPATH        = ROOTPATH / 'data'
RESPATH         = ROOTPATH / 'results'

LOG_FORMAT      = "%(asctime)s [%(levelname)-5.5s]  %(message)s"



