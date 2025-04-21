from trulens.core import TruSession
from trulens.dashboard import run_dashboard

if __name__ == "__main__":
    session = TruSession()
    run_dashboard(session)
