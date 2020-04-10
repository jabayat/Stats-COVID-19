#!/usr/bin/python
import sys
import COVID19_Utils

if __name__ == "__main__":
	# Parse arguments
	if len(sys.argv) < 2:
		COVID19_Utils.COVID19_loadDB('/Users/milko/Local/Data/Disease/COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/')
	else:
		COVID19_Utils.COVID19_loadDB(str(sys.argv)[1])
	
	# LoadDB('/Users/milko/Local/Data/Disease/COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/')
