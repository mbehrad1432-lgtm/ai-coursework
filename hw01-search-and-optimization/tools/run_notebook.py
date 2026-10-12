"""Execute this Python-only notebook sequentially without requiring Jupyter.

Captures stdout and Matplotlib PNG outputs; stops at the first failed cell.
This is a lightweight runner, not a Jupyter kernel or a general notebook runner.
"""
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'search_and_optimization.ipynb'

def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    namespace = {'__name__': '__main__'}
    os.chdir(ROOT)
    figures = ROOT / 'results/figures'
    figures.mkdir(parents=True, exist_ok=True)
    active_outputs = []
    image_count = 0

    def capture_figures(*args, **kwargs):
        nonlocal image_count
        for number in plt.get_fignums():
            fig = plt.figure(number)
            image_count += 1
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=110, bbox_inches='tight')
            data = buffer.getvalue()
            (figures / f'figure_{image_count:02d}.png').write_bytes(data)
            active_outputs.append({'output_type': 'display_data', 'metadata': {},
                                   'data': {'image/png': base64.b64encode(data).decode('ascii')}})
            plt.close(fig)

    plt.show = capture_figures
    count = 0
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        count += 1
        active_outputs = []
        stream = io.StringIO()
        try:
            with contextlib.redirect_stdout(stream):
                exec(compile(''.join(cell['source']), f'notebook-cell-{index}', 'exec'), namespace)
        except Exception as error:
            raise RuntimeError(f'Notebook cell {index} failed') from error
        output = stream.getvalue()
        cell['outputs'] = ([{'output_type': 'stream', 'name': 'stdout', 'text': output}]
                           if output else []) + active_outputs
        cell['execution_count'] = count
    notebook['nbformat'] = 4
    notebook['nbformat_minor'] = 4
    notebook['metadata']['language_info']['version'] = platform.python_version()
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    summary = {'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                'matplotlib': matplotlib.__version__},
               'execution': {'method': 'sequential Python cell execution',
                             'code_cells': count, 'figures': image_count},
               'maze': namespace['maze_summary'],
               'randomized_maze': {'successes': namespace['restart_successes'], 'runs': 100,
                                   'seeds': [0, 99], 'attempts_per_run': 20, 'p': 0.2},
               'one_dimensional': {'reference': namespace['reference_minimum'],
                                    'runs': namespace['one_dimensional_results']},
               'tsp': namespace['tsp_summary']}
    (ROOT / 'results/metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
