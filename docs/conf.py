# conf.py
from sphinx_pyproject import SphinxConfig

config = SphinxConfig("../pyproject.toml", globalns=globals())

# Now, variables like 'project', 'version', 'author' will be available
# from the loaded pyproject.toml data.
# You can also access other values through the 'config' object.
# For example: html_theme = config.html_theme if set in pyproject.toml
