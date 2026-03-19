import zipfile


class Loader(object):
    
    def __init__(self):
        pass
    
    """
    load_zip_iterator - allows for streaming data without spiking the RAM usage
    """
    @staticmethod
    def load_zip_iterator(filename: str):
        with zipfile.ZipFile(filename) as zref:
            target_file = next((f for f in zref.namelist() if f.endswith('.txt')))
            
            if not target_file:
                raise FileNotFoundError('No valid .txt file found in zip')
            
            with zref.open(target_file) as f:
                for line in f:
                    yield line.decode('utf-8', errors='ignore').strip()
    """
    load_zip - load entire contents of the ZIP archive (assuming it is only one file)    
    """
    @staticmethod
    def load_zip(filename: str) -> str:
        with zipfile.ZipFile(filename) as zref:
            target_file = next((f for f in zref.namelist() if not f.startswith('__')), zref.namelist()[0])
            return zref.read(target_file).decode('utf-8', errors='ignore').replace('\n', ' ')
            