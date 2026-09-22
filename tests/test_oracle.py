from dataclasses import replace
from flappy_fly.game.env import FlappyEnv
from flappy_fly.controllers.oracle import OracleController
from flappy_fly.controllers.random_controller import RandomController


def test_oracle_prediction_ceiling_cooldown():
    c = OracleController()
    s = replace(FlappyEnv().state, bird_y=350, bird_velocity_y=150, next_gap_center_y=280)
    assert c.act(s) == 1
    assert c.act(replace(s, elapsed_time=0.02)) == 0
    assert c.act(replace(s, elapsed_time=0.14)) == 1
    c.reset()
    assert c.act(replace(s, bird_y=30)) == 0


def test_oracle_better_than_random():
    scores = []
    for controller in (OracleController(), RandomController()):
        result = []
        for seed in range(20):
            env = FlappyEnv(seed)
            controller.reset(seed)
            for _ in range(1500):
                _, _, done, _ = env.step(controller.act(env.state))
                if done:
                    break
            result.append(env.score)
        scores.append(sum(result) / len(result))
    assert scores[0] >= 5
    assert scores[0] > scores[1] + 3


def test_dagger_teacher_queries_follow_actual_actions():
    c = OracleController()
    s = replace(FlappyEnv().state, bird_y=400, bird_velocity_y=150, next_gap_center_y=280)
    assert c.query(s) == c.query(s) == 1
    c.observe_action(s, 0)
    assert c.query(replace(s, elapsed_time=.02)) == 1
    c.observe_action(s, 1)
    assert c.query(replace(s, elapsed_time=.02)) == 0
