"""Plain-language labels for the learning dashboard. UI-only; never touches the brain."""

# What we artificially push into the fly's visual neurons (not real fly vision).
CHANNEL_GUIDE = {
    'LC4': {
        'title': 'Flap looks fatal',
        'role': 'High when TOO HIGH and flapping would hit the top.',
    },
    'LPLC2': {
        'title': 'Wait looks fatal',
        'role': 'High when TOO LOW and waiting would hit the bottom.',
    },
    'LC10a-L': {
        'title': 'Too low',
        'role': 'Strong → push FLAP (also boosts the controller directly).',
    },
    'LC10a-R': {
        'title': 'Too high',
        'role': 'Strong → brake FLAP so gravity can pull it down.',
    },
    'LPLC1-L': {
        'title': 'Falling',
        'role': 'On when the fly is moving downward.',
    },
    'LPLC1-R': {
        'title': 'Rising',
        'role': 'On when the fly is moving upward (after a flap).',
    },
}

OUTPUT_GUIDE = {
    'DNp01': {
        'title': 'Giant escape neuron',
        'role': 'Instinct mode flaps when this famous cell is loud.',
    },
    'descending_neuron': {
        'title': 'Body-command wires',
        'role': '1,314 outputs to the body. Brain mode reads all of them.',
    },
}


def wants_down(inputs):
    """Heuristic: fake eyes suggest falling (WAIT), not flapping up."""
    if not inputs:
        return False
    return inputs.get('LC4', 0) >= 0.35 or inputs.get('LC10a-R', 0) >= 0.25


def wants_up(inputs):
    """Heuristic: fake eyes suggest flapping up."""
    if not inputs:
        return False
    return inputs.get('LPLC2', 0) >= 0.35 or inputs.get('LC10a-L', 0) >= 0.25


def mismatch_warning(inputs, action):
    """Call out when sensing and motor choice disagree (common brain-mode failure)."""
    if action and wants_down(inputs) and not wants_up(inputs):
        return 'Mismatch: eyes say flap is dangerous, but it FLAPs anyway.'
    if (not action) and wants_up(inputs) and not wants_down(inputs):
        return 'Mismatch: eyes say waiting is fatal, but it WAITs anyway.'
    if wants_down(inputs) and not action:
        return 'Makes sense: flap looks fatal → WAIT → gravity pulls it down.'
    if wants_up(inputs) and action:
        return 'Makes sense: wait looks fatal → FLAP → goes up.'
    return ''


def narrate(inputs, action, probability, mode, threshold=0.5):
    """One short sentence a non-expert can skim."""
    if not inputs:
        return 'No brain attached — this mode does not inject fake vision.'
    bits = []
    if inputs.get('LC4', 0) >= 0.35:
        bits.append('next flap looks fatal')
    if inputs.get('LPLC2', 0) >= 0.35:
        bits.append('waiting looks fatal')
    if inputs.get('LC10a-L', 0) >= 0.25:
        bits.append('below the hole')
    if inputs.get('LC10a-R', 0) >= 0.25:
        bits.append('above the hole')
    if inputs.get('LPLC1-L', 0) >= 0.35:
        bits.append('falling')
    if inputs.get('LPLC1-R', 0) >= 0.35:
        bits.append('rising')
    scene = ', '.join(bits) if bits else 'neither action looks immediately fatal'
    if mode == 'instinct':
        decision = 'FLAP (DNp01 loud)' if action else 'WAIT (DNp01 quiet)'
    elif probability is None:
        decision = 'FLAP' if action else 'WAIT'
    elif action:
        decision = f'FLAP (score {probability:.0%} ≥ {threshold:.0%})'
    else:
        decision = f'WAIT (score {probability:.0%} < {threshold:.0%})'
    return f'Sees: {scene}. Decides: {decision}.'
