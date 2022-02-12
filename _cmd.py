import os
import click

from src.models import dispatch
from src.models import output_graph
from src.data import format_data

@click.group()
def cli():
    pass

@click.command()
@click.option('-o', '--ori', help='time series path', required=True)
@click.option('-d', '--dest', help='time series result destination', required=True)
def train_models(ori, dest):
    dispatch.dispatch(ori, dest)

@click.command()
@click.option('-d', '--dest', help='time series result destination', required=True)
def generate_graph(dest):
    output_graph.create_graph(dest)

cli.add_command(train_models)
cli.add_command(generate_graph)

if __name__ == "__main__":
    cli()