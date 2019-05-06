from pathlib import Path

ROOT = Path('/home/cgsdfc/SavedModels/HRED-VHRED/LSDSCC/VHRED')

if __name__ == '__main__':
    files = list(ROOT.iterdir())
    for file in files:
        if file.name.startswith('Opensub'):
            file.rename(file.with_name(file.name.replace('Opensub', 'LSDSCC')))
