from policies.policy import Policy

from math_utils import tuple_addition

def generate_episode(policy: Policy, grid_pos: tuple[int, int]) -> list[tuple[tuple[int, int], int], tuple[int, int]]:
    episode: list[tuple[tuple[int, int], int], tuple[int, int]] = list()

    pos = grid_pos

    while (True):
        act = policy.move(pos)
        next_pos = tuple_addition(pos, act)

        episode.append((pos, act, policy.grid[next_pos].reward))

        if (policy.grid[pos].is_terminal):
            break

        pos = next_pos  

    return episode