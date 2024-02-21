import os
from trulens_eval import Tru
from dotenv import load_dotenv
load_dotenv()

tru = Tru(database_url=os.environ['TRULENS_DB_URL'])
tru.run_dashboard()