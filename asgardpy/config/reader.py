from ruamel.yaml import YAML
from pathlib import Path

class Configuration(object):
    """Configuration class containing all variables

    Args:
        filename (str): path to the yaml configuration file
    """
    def __init__(self,filename=None):
        if filename is not None:
            self.read_file(filename)
    
    def read_file(self,filename):
        """read configuration from a file

        Args:
            filename (str): path to the yaml configuration file to read
        """
        yaml = YAML()
        self.config = yaml.load(Path(filename))
    
    def write_file(self,filename):
        """write the configuration to a file

        Args:
            filename (str): path to the yaml configuration file to write
        """
        yaml = YAML()
        yaml.dump(self.config,Path(filename))
    
    def load_template(self):
        """load template configuration file.
        """
        yaml = YAML()
        self.default_config = yaml.load(Path("template.yaml"))
    
    def validate(self):
        """Validate configuration 
        """