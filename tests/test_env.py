from flappy_fly.game.env import FlappyEnv


def test_seed_and_actions_reproduce():
    a, b = FlappyEnv(17), FlappyEnv(17)
    for t in range(300):
        assert a.step(int(t % 28 == 0)) == b.step(int(t % 28 == 0))
    assert a.reset() == b.reset()
    assert a.pipes == b.pipes


def test_flap_gravity_collision_and_reset():
    env = FlappyEnv()
    state, *_ = env.step(1)
    assert state.bird_velocity_y < 0
    second, *_ = env.step(0)
    assert second.bird_velocity_y > state.bird_velocity_y
    for _ in range(200):
        state, _, done, _ = env.step(0)
        if done:
            break
    assert not state.alive
    assert env.step(0)[1] == 0
    state = env.reset()
    assert state.alive and state.score == 0 and state.elapsed_time == 0


def test_pass_counts_once_and_pipe_collides():
    env = FlappyEnv()
    g = env.config.game
    pipe = env.pipes[0]
    pipe.x = g.fly_x - g.fly_radius - g.pipe_width + 1
    pipe.gap_center = env.y
    assert env.step(0)[0].score == 1
    assert env.step(0)[0].score == 1
    env.reset()
    env.pipes[0].x = g.fly_x
    env.y = 30
    assert env.step(0)[2]
