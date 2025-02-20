from policies.policy import Policy


def evaluate_policy(policy: Policy, possible_rewards = None, gamma: float = 0.8, theta: float = 0.001):
    # If not provided with a list of possible rewards, initialize a list for the entire grid with value 0.
    if possible_rewards is None:
        possible_rewards = {state: 0 for state in policy.grid.keys()}

    delta = theta + 1
    while delta > theta:
        delta = 0

        # Iterate for each cell in the grid.
        for s in policy.grid.keys():
            v = 0
            # Get the cell associated with the coordinate (s).
            cell = policy.grid[s]
            if cell.is_terminal:
                continue

            # Iterate over each action possible in this cell.
            for action in policy.movement[s]:
                # Combine coordinate and action to create the new position following the action.
                next_state = (action[0] + s[0], action[1] + s[1])
                # In this simulation the actions are deterministic therefore it is multiplied by 1 instead of a probability.
                v += policy.movement[s][action] * (policy.grid[next_state].reward + gamma * possible_rewards[next_state])

            # Make the difference a positive number and replace it with delta if it is bigger.
            delta = max(delta, abs(v - possible_rewards[s]))
            # Add v as the possible reward for the coordinate s.
            possible_rewards[s] = v

    return possible_rewards


def improve_policy(policy: Policy, gamma: float = 0.8):
    policy_stable = False

    while not policy_stable:
        policy_stable = True
        possible_rewards = evaluate_policy(policy)

        # Iterate for each cell in the grid.
        for s in policy.grid.keys():
            # Get the cell associated with the coordinate (s).
            cell = policy.grid[s]
            if cell.is_terminal:
                continue
            old_action = policy.movement[s]
            eval_actions = {}

            # Iterate over each action possible in this cell.
            for action in policy.movement[s]:
                next_state = (action[0] + s[0], action[1] + s[1])
                next_reward = policy.grid[next_state].reward + gamma * possible_rewards[next_state]
                eval_actions[action] = next_reward

            best_actions = [action for action, reward in eval_actions.items() if reward == max(eval_actions.values())]
            action_prob = 1 / len(best_actions)
            new_actions = {}

            for action in best_actions:
                new_actions[action] = action_prob

            if old_action != new_actions:
                policy.movement[s] = new_actions
                policy_stable = False

        if policy_stable:
            return possible_rewards
