import sys
import pytest
from flappy_fly.training.evaluate import evaluate
from flappy_fly.config import Config


def test_same_seeds_and_caps(tmp_path):
    report = evaluate(episodes=3, seed=501, max_steps=20, modes=['oracle', 'random'], output=tmp_path/'eval.json')
    assert report['seeds'] == [501, 502, 503]
    for result in report['results'].values():
        assert [r['seed'] for r in result['episodes']] == report['seeds']
        assert all(r['survival_time'] <= 20 * Config().physics.dt for r in result['episodes'])
    assert report['results']['oracle']['summary']['truncated_episodes'] == 3


def test_evaluation_fails_early_without_model(tmp_path):
    with pytest.raises(ValueError, match='requires --model'):
        evaluate(episodes=1, max_steps=1, output=tmp_path/'eval.json')
    assert not (tmp_path/'eval.json').exists()
