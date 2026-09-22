from .state import FLAP


def integrate(y, velocity, action, config):
    if action == FLAP:
        velocity = config.flap_velocity
    velocity = min(velocity + config.gravity * config.dt, config.max_fall_speed)
    return y + velocity * config.dt, velocity


def circle_rectangle(cx, cy, radius, x, y, width, height):
    near_x = max(x, min(cx, x + width))
    near_y = max(y, min(cy, y + height))
    return (cx - near_x) ** 2 + (cy - near_y) ** 2 <= radius ** 2
