from typing import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker
import seaborn as sns

###
# DataHubIo - Utilities for reading and displaying data from the
#             https://github.com/datasets/covid-19.git
#             Repository.
###

class Dataset:
	'''
	This class implements the following functionality:
	- Load the dataset.
	- Transform the dataset.
	- Plot the data.
	'''

	population_latest = '2018'
	population_file = 'API_SP.POP.TOTL_DS2_en_csv_v2_936048.csv'
	dataset_site = 'https://raw.githubusercontent.com/datasets/covid-19/master/data/'
	dataset_name = 'countries-aggregated.csv'
	names_match = 'Population-COVID-country-name-match.csv'
	palette = [
		'#045275', '#089099', '#7CCBA2', '#FCDE9C', '#DC3977', '#7C1D6F', '#7C7D3F'
	]

	def __init__(self, file: str = None):
		'''
		Load the dataset and the population.

		param: file: Provide file path to load from file.
		'''

		# Load from host
		if file is None:
			self.dataset_ = pd.read_csv(
				self.dataset_site + self.dataset_name,
				parse_dates=['Date']
			)

		# Load from file
		else:
			self.dataset_ = pd.read_csv(
				file,
				parse_dates=['Date']
			)

		# Load population
		self.population_ = pd.read_csv(
			self.population_file,
			usecols=['Country Name', 'Country Code', self.population_latest],
			header=2
		)

		# Adjust country names
		country_match = \
			pd.read_csv(self.names_match)\
			.set_index('Country')\
			.to_dict()['Country Name']
		self.dataset_['Country'] = \
			self.dataset_['Country'].apply(
				lambda x:
					country_match[x] if x in country_match.keys()
					else x
			)

		# Merge with population
		self.dataset_ = \
			self.dataset_.merge(
				self.population_,
				how='inner',
				left_on='Country',
				right_on='Country Name'
			).drop(columns='Country Name')\
		 	.dropna()

		# Collect countries and codes
		self.country_names = self.dataset_['Country'].unique()
		self.country_codes = self.dataset_['Country Code'].unique()

	def get_country_data(self, country: str):
		'''
		Return country dataset

		param country: str, The name or ISO code of the country
		returns pandas.DataFrame: The country dataset, or None
		'''

		# Try country code
		if country in self.country_codes:
			df = self.dataset_[self.dataset_['Country Code'] == country]

		# Try country name
		elif country in self.country_names:
			df = self.dataset_[self.dataset_['Country'] == country]

		# Country not found
		else:
			print("Country {} not found.".format(country))
			return None													# ==>

		# Prepare dataset
		country = df['Country'].unique()[0]
		df = df.sort_values('Date', ignore_index=True)
		df['Active'] = df['Confirmed'] - ( df['Recovered'] + df['Deaths'] )
		df.set_index('Date', inplace=True)

		return df														# ==>

	def get_countries_data(self, countries: [str]):
		'''
		Return data for a set of countries

		param countries: [str], List of countries
		returns pandas DataFrame, Datases
		'''

		# Match countries
		matched = []
		country_matcher = dict(zip(self.country_names, self.country_codes))
		for country in countries:
			if country in self.country_codes:
				matched.append(country)
			elif country in self.country_names:
				matched.append(country_matcher[country])
		if len(matched) == 0:
			print("No countries matched.")
			return None													# ==>

		# Prepare dataset
		df = self.dataset_[self.dataset_['Country Code'].isin(matched)].copy()
		df['Active'] = df['Confirmed'] - (df['Recovered'] + df['Deaths'])
		df['Confirmed'] = df['Confirmed'] / df[self.population_latest] * 10000
		df['Recovered'] = df['Recovered'] / df[self.population_latest] * 10000
		df['Active'] = df['Active'] / df[self.population_latest] * 10000
		df['Deaths'] = df['Deaths'] / df[self.population_latest] * 10000
		df.drop(columns=['Country Code', self.population_latest], inplace=True)

		return df														# ==>

	def plot_country_cases(self, country: str):
		'''
		Plot country statistics

		param country: str, The name or ISO code of the country
		'''

		# Get country dataset
		df = self.get_country_data(country)
		if df is None:
			return														# ==>

		# Prepare dataset
		country = df['Country'].unique()[0]
		df.drop(
			columns=['Country', 'Country Code', self.population_latest],
			inplace=True
		)

		# Set palette
		palette = ['#7C1D6F', '#7CCBA2', '#DC3977', '#FCDE9C']
		columns = ['Confirmed', 'Recovered', 'Deaths', 'Active']
		colors = dict(zip(columns, palette))

		# Set style
		plt.style.use('fivethirtyeight')

		# Generate plot
		plot = df.plot(
			kind='line',
			figsize=(12,8),
			color=list(colors.values()), alpha=0.7,
			linewidth=5,
			legend=False
		)

		# Set axis
		plot.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
		plot.grid(color='#d4d4d4')
		plot.set_xlabel('Date')
		plot.set_ylabel('# of Cases')

		# Assign colors
		for column in list(colors.keys()):
			plot.text(
				x = df.index[-1], y = df[column].max(),
				color = colors[column],
				s = column, weight = 'bold'
			)

		# Add labels
		plot.text(
			x = df.index[1], y = int(df.max().max()),
			s = country, fontsize = 23, weight = 'bold', alpha = 0.75
		)

	def plot_country_daily_cases(self, country: str):
		'''
		Plot daily country statistics

		param country: str, The name or ISO code of the country
		'''

		# Get country dataset
		df = self.get_country_data(country)
		if df is None:
			return														# ==>

		# Prepare dataset
		country = df['Country'].unique()[0]
		df = \
			df.drop(columns=['Country', 'Country Code', self.population_latest])\
			  .diff() \
			  .fillna(0)
		df.reset_index(inplace=True)
		df['Days'] = (df['Date'] - df['Date'].min()) / np.timedelta64(1, 'D')

		# Set palette
		palette = ['#7C1D6F', '#7CCBA2', '#DC3977', '#FCDE9C']
		columns = ['Confirmed', 'Recovered', 'Deaths', 'Active']
		colors = dict(zip(columns, palette))

		# Set facets
		facets = {
			'confirmed': {
				'column': 'Confirmed',
				'color': 'blue',
				'label': "Confirmed"
			},
			'active': {
				'column': 'Active',
				'color': 'orange',
				'label': "Active"
			},
			'recovered': {
				'column': 'Recovered',
				'color': 'green',
				'label': "Recovered"
			},
			'dead': {
				'column': 'Deaths',
				'color': 'red',
				'label': "Deaths"
			}
		}

		alpha = 0.7
		order = 3
		dots = 10

		# Set style
		plt.style.use('fivethirtyeight')

		# Plot
		g = sns.FacetGrid(data=df, height=8, aspect=1.2)
		for key in facets.keys():
			g.map(
				sns.regplot,
				data=df, x='Days', y=facets[key]['column'],
				order=order,
				scatter_kws=dict(alpha=alpha, s=dots, color=facets[key]['color']),
				line_kws=dict(alpha=alpha, color=facets[key]['color'], linewidth=3),
				label=facets[key]['label']
			)

		# Legend
		plt.legend()

	def compare_countries_confirmed(self, countries: [str]):
		'''
		Compare country statistics

		param countries: [str], List of countries to compare
		'''

		# Get countries data
		df = self.get_countries_data(countries)
		if df is None:
			return														# ==>

		# Save countries
		countries = df['Country'].unique()

		# Prepare dataset
		df = df.pivot(index='Date', columns='Country', values='Confirmed')
		dataset = df.reset_index('Date')
		dataset.set_index(['Date'], inplace=True)

		# Prepare country data
		colors = dict(zip(countries, sns.color_palette('muted', len(countries)).as_hex()))

		# Set style
		plt.style.use('fivethirtyeight')

		# Generate plot
		plot = dataset.plot(figsize=(12,8), color=list(colors.values()), linewidth=5, legend=False)
		plot.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
		plot.grid(color='#d4d4d4')
		plot.set_xlabel('Date')
		plot.set_ylabel('# of Cases x 100k')

		# Assign colors
		for country in list(colors.keys()):
			plot.text(
				x = dataset.index[-1], y = dataset[country].max(),
				color = colors[country],
				s = country, weight = 'bold'
			)

		# Add labels
		plot.text(
			x = df.index[1], y = int(df.max().max()),
			s = "Confirmed Cases by Country", fontsize = 23, weight = 'bold', alpha = 0.75
		)

	def compare_countries_active(self, countries: [str]):
		'''
		Compare country statistics

		param countries: [str], List of countries to compare
		'''

		# Get countries data
		df = self.get_countries_data(countries)
		if df is None:
			return														# ==>

		# Save countries
		countries = df['Country'].unique()

		# Prepare dataset
		df = df.pivot(index='Date', columns='Country', values='Active')
		dataset = df.reset_index('Date')
		dataset.set_index(['Date'], inplace=True)

		# Prepare country data
		colors = dict(zip(countries, sns.color_palette('muted', len(countries)).as_hex()))

		# Set style
		plt.style.use('fivethirtyeight')

		# Generate plot
		plot = dataset.plot(figsize=(12,8), color=list(colors.values()), linewidth=5, legend=False)
		plot.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
		plot.grid(color='#d4d4d4')
		plot.set_xlabel('Date')
		plot.set_ylabel('# of Cases x 100k')

		# Assign colors
		for country in list(colors.keys()):
			plot.text(
				x = dataset.index[-1], y = dataset[country].max(),
				color = colors[country],
				s = country, weight = 'bold'
			)

		# Add labels
		plot.text(
			x = df.index[1], y = int(df.max().max()),
			s = "Active Cases by Country", fontsize = 23, weight = 'bold', alpha = 0.75
		)

	def compare_countries_recovered(self, countries: [str]):
		'''
		Compare country statistics

		param countries: [str], List of countries to compare
		'''

		# Get countries data
		df = self.get_countries_data(countries)
		if df is None:
			return														# ==>

		# Save countries
		countries = df['Country'].unique()

		# Prepare dataset
		df = df.pivot(index='Date', columns='Country', values='Recovered')
		dataset = df.reset_index('Date')
		dataset.set_index(['Date'], inplace=True)

		# Prepare country data
		colors = dict(zip(countries, sns.color_palette('muted', len(countries)).as_hex()))

		# Set style
		plt.style.use('fivethirtyeight')

		# Generate plot
		plot = dataset.plot(figsize=(12,8), color=list(colors.values()), linewidth=5, legend=False)
		plot.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
		plot.grid(color='#d4d4d4')
		plot.set_xlabel('Date')
		plot.set_ylabel('# of Cases x 100k')

		# Assign colors
		for country in list(colors.keys()):
			plot.text(
				x = dataset.index[-1], y = dataset[country].max(),
				color = colors[country],
				s = country, weight = 'bold'
			)

		# Add labels
		plot.text(
			x = df.index[1], y = int(df.max().max()),
			s = "Recovered Cases by Country", fontsize = 23, weight = 'bold', alpha = 0.75
		)

	def compare_countries_deaths(self, countries: [str]):
		'''
		Compare country statistics

		param countries: [str], List of countries to compare
		'''

		# Get countries data
		df = self.get_countries_data(countries)
		if df is None:
			return														# ==>

		# Save countries
		countries = df['Country'].unique()

		# Prepare dataset
		df = df.pivot(index='Date', columns='Country', values='Deaths')
		dataset = df.reset_index('Date')
		dataset.set_index(['Date'], inplace=True)

		# Prepare country data
		colors = dict(zip(countries, sns.color_palette('muted', len(countries)).as_hex()))

		# Set style
		plt.style.use('fivethirtyeight')

		# Generate plot
		plot = dataset.plot(figsize=(12,8), color=list(colors.values()), linewidth=5, legend=False)
		plot.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
		plot.grid(color='#d4d4d4')
		plot.set_xlabel('Date')
		plot.set_ylabel('# of Cases x 100k')

		# Assign colors
		for country in list(colors.keys()):
			plot.text(
				x = dataset.index[-1], y = dataset[country].max(),
				color = colors[country],
				s = country, weight = 'bold'
			)

		# Add labels
		plot.text(
			x = df.index[1], y = int(df.max().max()),
			s = "Death Cases by Country", fontsize = 23, weight = 'bold', alpha = 0.75
		)
