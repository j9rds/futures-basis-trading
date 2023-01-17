# Project Name
This project scrapes data from https://coinalyze.net/bitcoin/basis/ and saves it to a json or csv file. 

## Getting Started
The following packages are used in this project:
- requests
- BeautifulSoup
- pandas
- time
- json

## Usage
The main function, `scrape()`, takes in two arguments: `list_of_cols` and `save_as`.
- `list_of_cols` is a list of columns to be included in the output.
- `save_as` is a string that specifies the format of the output file. It can either be `json` or `csv`.

### Constants
The following constants are used in this project:
- `list_of_cols` is a list of columns to be included in the output.
- `every_n_mins` is the waiting time (in minutes) between each scrape.
- `out_file` is the name of the output file.
- `save_as` is a string that specifies the format of the output file. It can either be `json` or `csv`.


