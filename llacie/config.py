import os
from dotenv import dotenv_values
from click import BadParameter
from collections import UserDict

from .utils import echo_warn

class ConfigError(BadParameter):
    pass

class Config(UserDict):
    """Holds configuration for the LLaCIE command line app.
    
    Most values are loaded from .env files: example default values are in
    .env.example, and they can be copied to .env and overridden.
    
    This object offers a dictionary-like interface, except that missing
    values return None rather than raising a KeyError.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data.update({
            **dotenv_values(".env.example"),
            **dotenv_values(".env"),
            **os.environ
        })

        # Set os.environ if we have a HF_TOKEN in a config file, since the `transformers` library
        # will read it from there and not our config files
        if self.data.get('HF_TOKEN') is not None and os.environ.get('HF_TOKEN') is None:
            os.environ['HF_TOKEN'] = self.data.get('HF_TOKEN')

        if self.data.get('PG_URI') is None:
            self.data['PG_URI'] = self._get_pg_connection_string()
        
        if self.data.get('EDW_URI') is None:
            self.data['EDW_URI'] = self._get_edw_connection_string()


    def __missing__(self, key):
        return None
    

    def get(self, key, default=None):
        if key in self.data:
            return self.data[key]
        return default


    def _get_pg_connection_string(self):
        pg_user = self['PG_USER']
        pg_password = self['PG_PASS']
        pg_host = self['PG_HOST']
        pg_dbname = self['PG_DBNAME']

        if pg_host is None or pg_dbname is None:
            raise ConfigError("PG_HOST and PG_DBNAME need to be configured")

        if pg_user is None or pg_password is None or pg_password == '':
            try:
                with open(os.environ['HOME'] + "/.pgpass", "r") as f:
                    for line in f.read().split('\n'):
                        fields = line.strip().split(':')
                        if fields[0] == pg_host and fields[2] == pg_dbname:
                            pg_user = fields[3]
                            pg_password = fields[4]
            except IOError:
                echo_warn("Tried to load postgres login info from ~/.pgpass but it wasn't readable")
        
        if pg_user is None or pg_password is None:
            raise ConfigError("Could not find postgres login information")
        
        return f"postgresql://{pg_user}:{pg_password}@{pg_host}/{pg_dbname}"
    

    def _get_edw_connection_string(self):
        edw_user = self['EDW_USER']
        edw_pass = self['EDW_PASS']
        edw_host = self['EDW_HOST']

        if edw_pass is None:
            try:
                with open(os.environ['HOME'] + "/.edw_env", "r") as f:
                    edw_pass = f.read().split('"')[1]
            except IOError:
                pass

        if edw_user is None or edw_pass is None or edw_host is None:
            return None
        
        return f"mssql+pyodbc://{edw_user}:{edw_pass}@{edw_host}"
