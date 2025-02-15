from policies.policy import Policy


def evaluate_policy(policy: Policy, possible_rewards = None, gamma: float = 1, theta: float = 0.0001):
    if possible_rewards is None:
        possible_rewards = {state: 0 for state in policy.grid.keys()}

    while True:
        delta = 0

        for s in policy.grid.keys():
            v = 0
            cell = policy.grid[s]
            if cell.is_terminal:
                continue

            for action in policy.movement[s]:
                next_state = (action[0] + s[0], action[1] + s[1])
                v += policy.movement[s][action] * (policy.grid[next_state].reward + gamma * possible_rewards[next_state])

            delta = max(delta, abs(v - possible_rewards[s]))
            possible_rewards[s] = v

        if delta < theta:
            break

    return possible_rewards



















    # diff = theta + 1
    # while diff > theta:
    #     diff = 0
    #     grid = policy.grid
    #     for state in grid.keys():
    #         if grid[state].is_terminal:
    #             possible_rewards[state] = 0
    #             continue
    #
    #         current_reward = possible_rewards[state]
    #         possible_reward = 0
    #
    #         for action in policy.movement[state]:
    #             action_state = (action[0] + state[0], action[1] + state[1])
    #             for sub_action in policy.movement[action_state]:
    #                 sub_action_state = (sub_action[0] + action_state[0], sub_action[1] + action_state[1])
    #                 possible_reward += policy.movement[state][action] * 1 * (policy.grid[action_state].reward + gamma * possible_rewards[sub_action_state])
    #
    #         possible_rewards[state] = possible_reward
    #         diff = max(diff, abs(possible_reward - current_reward))
    #
    # return possible_rewards
