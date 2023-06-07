import pandas as pd
import logging
from ydata_profiling import ProfileReport

logging.basicConfig(filename='example.log', level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')

data = pd.read_csv('data/blue_medium_data_task1.csv')
profile = ProfileReport(data, title = 'profiling report')
profile.to_file("ML_ppreport.html")


