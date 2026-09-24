"""Orderly shutdown for the standalone edge HTTP server."""

import logging
import signal

logger = logging.getLogger("jetson-app")


def serve(app, runtime, **options):
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)

    def terminate(_signum, _frame):
        raise SystemExit(0)

    def draining(_signum, _frame):
        logger.warning("Shutdown is already draining capture and inference workers; please wait")

    signal.signal(signal.SIGTERM, terminate)
    try:
        app.run(**options)
    except KeyboardInterrupt:
        pass
    finally:
        # A second Ctrl+C during cleanup used to interrupt joins and leave
        # native decoder threads running as Python finalized their objects.
        signal.signal(signal.SIGINT, draining)
        signal.signal(signal.SIGTERM, draining)
        try:
            runtime.close()
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            signal.signal(signal.SIGINT, previous_int)
