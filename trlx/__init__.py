"""trlx library.

TrlxError identifies expected user-facing failures. Both expected errors and
unexpected exceptions retain their cause and operation evidence through the
shared failure reporter; unexpected failures retain technical tracebacks.
"""


from dataset.failures import Error


class TrlxError(Error):
    pass
