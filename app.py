import sys
from streamlit import cli as stcli

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "main/streamlit/measurement_script/measurement_script.py"]
    sys.exit(stcli.main())
    #CMD streamlit run --server.port 8080 --server.enableCORS false app.py
