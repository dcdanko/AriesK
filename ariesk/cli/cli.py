import click

from .cli_dev import dev_cli
from .cli_stats import stats_cli
from .cli_build import build_cli
from .cli_search import search_cli


@click.group()
def main():
    pass


main.add_command(dev_cli)
main.add_command(stats_cli)
main.add_command(build_cli)
main.add_command(search_cli)
