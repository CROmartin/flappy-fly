from .encoder import CHANNELS


def resolve_groups(brain):
    groups = {}
    for name in CHANNELS:
        parts = name.split('-')
        groups[name] = brain.cells([parts[0]], side=parts[1] if len(parts) == 2 else None)
    for name in ('LC4', 'LPLC2'):
        for side in ('L', 'R'):
            groups[f'{name}-{side}'] = brain.cells([name], side=side)
    groups['DNp01'] = brain.cells(['DNp01'])
    groups['descending_neuron'] = brain.cells(['descending_neuron'])
    empty = [name for name, idx in groups.items() if len(idx) == 0]
    if empty:
        raise RuntimeError(f'Required connectome populations missing: {empty}')
    return groups
