import os
import click

from src.models import dispatch
from src.models import output_graph
from src.data import format_data
from src.data import create_datasets

@click.group()
def cli():
    pass

@click.command()
@click.option('-o', '--ori', help='forecast time series', required=True)
@click.option('-d', '--dest', help='time series path to use in modelling', required=True)
@click.option('-b', '--buoy_path', help='real buoy data path', required=True)
@click.option('-n', '--name', help='name of buoy location with processed data', required=True)
@click.option('-l', '--lag', help='number of days for prediction',required=True, type=int)
def create_data(ori, dest, buoy_path, name, lag):
    create_datasets.dispatch(ori, dest, buoy_path, name, lag)

@click.command()
@click.option('-o', '--ori', help='time series path', required=True)
@click.option('-d', '--dest', help='time series result destination', required=True)
@click.option('-n', '--name', help='name of buoy location with processed data', required=True)
def train_models(ori, dest, name):
    dispatch.dispatch(ori, dest, name)

@click.command()
@click.option('-d', '--dest', help='time series result destination', required=True)
def generate_graph(dest):
    output_graph.create_graph(dest)

cli.add_command(create_data)
cli.add_command(train_models)
cli.add_command(generate_graph)

if __name__ == "__main__":
    cli()
