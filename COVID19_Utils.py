import os
import glob
import pandas as pd
import numpy as np
from arango import ArangoClient

def COVID19_loadDB(path, ip='http://localhost:8529', dbname='COVID-19', col_name='daily'):
    '''
    Load database with all reports in provided directory.
    
    INPUT
        path: Directory containing reports.
        dbname: Database name.
    '''
    
    # Get client and system database.
    client = ArangoClient(hosts=ip)
    sys_db = client.db(username='root', password='HHeXW)YE3rm8cnPw')
    
    # Create/open database.
    if not sys_db.has_database(dbname):
        sys_db.create_database(dbname)
    db = client.db(dbname, username='root', password='HHeXW)YE3rm8cnPw')
    
    # Create/truncate collection.
    if db.has_collection(col_name):
        collection = db.collection(col_name)
        collection.truncate()
    else:
        collection = db.create_collection(col_name)
    
    # Iterate reports.
    for file in glob.glob(path + "*.csv"):
        
        # Extract date.
        month, day, year = os.path.basename(file)[:10].split('-')
        date = '-'.join([year, month, day])
        print(date)
        
        # Load report.
        report = pd.read_csv(file)
        
        # Rename columns.
        report.rename(
            columns={
                'Country/Region': 'Country',
                'Province/State': 'Province'
            }, inplace=True)

        report.rename(
            columns={
                'Country_Region': 'Country',
                'Province_State': 'Province'
            }, inplace=True)
        
        # Group by country.
        report = report.groupby('Country').sum()
        
        # Add date.
        report['Date']  = date

        # Set computed active cases
        report['Current'] = report['Confirmed'] - (report['Recovered'] + report['Deaths'])

        # Replace NaN with zero.
        report.fillna(0, inplace=True)
        report.replace(to_replace=float('inf'), value=0, inplace=True)
        
        # Convert index to column.
        report.reset_index(drop=False, inplace=True)
        
        # Write to database.
        collection.insert_many(report.to_dict(orient='records'), silent=True)
        
def COVID19_country2CSV(country, ip='http://localhost:8529', dbname='COVID-19', col_name='daily'):
    '''
    Save a CSV file with provided country data.

    INPUT
        country: Name of country.
    '''
    
    # Get connection to ArangoDB collection.
    client = ArangoClient(hosts=ip)
    db = client.db(dbname, username='root', password='HHeXW)YE3rm8cnPw')

    # Get data from database.
    cursor = db.aql.execute(
        "FOR doc IN daily FILTER doc.Country == @country RETURN doc",
        bind_vars={'country': country}
    )
    
    # Get DataFrame.
    df = pd.DataFrame.from_records(
        [document for document in cursor],
        index=['Date'],
        exclude=['_key', '_id', '_rev', 'Country'])

    # Sort data frame by date in ascending order.
    df.sort_index(inplace=True)

    # Add delta values.
    df = pd.concat(
        [
            df,
            df[['Confirmed', 'Current', 'Recovered', 'Deaths']]
                .diff(axis=0)
                .rename(columns={
                    'Confirmed': 'NewConfirmed',
                    'Current':   'NewCurrent',
                    'Recovered': 'NewRecovered',
                    'Deaths':    'NewDeaths'
                })
        ],
        axis=1
    )

    # Normalise values.
    df.fillna(0, inplace=True)
    df.replace(to_replace=float('inf'), value=0, inplace=True)
    
    # Add growth rates.
    df = pd.concat(
        [
            df,
            pd.DataFrame({
                'GrConfirmed': df['NewConfirmed'] / (df['Confirmed'] - df['NewConfirmed']),
                'GrCurrent': df['NewCurrent'] / (df['Current'] - df['NewCurrent']),
                'GrRecovered': df['NewRecovered'] / (df['Recovered'] - df['NewRecovered']),
                'GrDeaths': df['NewDeaths'] / (df['Deaths'] - df['NewDeaths'])
            })
        ],
        axis=1
    )

    # Normalise values.
    df.fillna(0, inplace=True)
    df.replace(to_replace=float('inf'), value=0, inplace=True)

    # Save file.
    df.to_csv(f"{country}.csv")

def COVID19_country2DF(country, ip='http://localhost:8529', dbname='COVID-19', col_name='daily'):
    '''
    Return a DataFrame with provided country data.

    INPUT
        country: Name of country.
    
    OUTPUT
        DataFrame: Country data with delta values.
    '''
    
    # Get connection to ArangoDB collection.
    client = ArangoClient(hosts=ip)
    db = client.db(dbname, username='root', password='HHeXW)YE3rm8cnPw')

    # Get data from database.
    cursor = db.aql.execute(
        "FOR doc IN daily FILTER doc.Country == @country RETURN doc",
        bind_vars={'country': country}
    )
    
    # Get DataFrame.
    df = pd.DataFrame.from_records(
        [document for document in cursor],
        index=['Date'],
        exclude=['_key', '_id', '_rev', 'Country'])

    # Sort data frame by date in ascending order.
    df.sort_index(inplace=True)

    # Add delta values.
    df = pd.concat(
        [
            df,
            df[['Confirmed', 'Current', 'Recovered', 'Deaths']]
                .diff(axis=0)
                .rename(columns={
                    'Confirmed': 'NewConfirmed',
                    'Current':   'NewCurrent',
                    'Recovered': 'NewRecovered',
                    'Deaths':    'NewDeaths'
                })
        ],
        axis=1
    )

    # Normalise values.
    df.fillna(0, inplace=True)
    df.replace(to_replace=float('inf'), value=0, inplace=True)
    
    # Add growth rates.
    df = pd.concat(
        [
            df,
            pd.DataFrame({
                'GrConfirmed': df['NewConfirmed'] / (df['Confirmed'] - df['NewConfirmed']),
                'GrCurrent': df['NewCurrent'] / (df['Current'] - df['NewCurrent']),
                'GrRecovered': df['NewRecovered'] / (df['Recovered'] - df['NewRecovered']),
                'GrDeaths': df['NewDeaths'] / (df['Deaths'] - df['NewDeaths'])
            })
        ],
        axis=1
    )

    # Normalise values.
    df.fillna(0, inplace=True)
    df.replace(to_replace=float('inf'), value=0, inplace=True)
    
    return df
