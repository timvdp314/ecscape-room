from cell import Cell
from policies.policy import Policy

import matplotlib.pyplot as plt

def evaluate_policy(policy: Policy, possible_rewards: dict[(int, int)] = None, gamma: float = 0.8, theta: float = 0.001) -> dict[(int, int)]:
    # If not provided with a list of possible rewards, initialize a list for the entire grid with value 0.
    if possible_rewards is None:
        possible_rewards = {state: 0 for state in policy.grid.keys()}

    delta: float = theta + 1
    while delta > theta:
        delta = 0

        # Iterate for each cell in the grid.
        for s in policy.grid.keys():
            v: float = 0
            # Get the cell associated with the coordinate (s).
            cell: Cell = policy.grid[s]
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


def improve_policy(policy: Policy, gamma: float = 0.8) -> dict[(int, int)]:
    policy_stable = False

    # While the policy is not stable, it is not optimal yet.
    while not policy_stable:
        # The policy is set to stable so that if no value is changed the loop stops.
        policy_stable = True
        possible_rewards: dict[(int, int)] = evaluate_policy(policy)

        # Iterate for each cell in the grid.
        for s in policy.grid.keys():
            # Get the cell associated with the coordinate (s).
            cell: Cell = policy.grid[s]
            if cell.is_terminal:
                continue

            # Save the current actions.
            old_actions: dict[(int, int)] = policy.movement[s]
            # A dictionary of the new evaluated actions.
            eval_actions: dict[(int, int)] = {}

            # Iterate over each action possible in this cell.
            for action in policy.movement[s]:
                # Create the new position based on the action and the current position.
                next_state: (int, int) = (action[0] + s[0], action[1] + s[1])
                # Calculate the new reward given the action.
                next_reward: float = policy.grid[next_state].reward + gamma * possible_rewards[next_state]
                eval_actions[action] = next_reward

            # Get the best actions out of the new actions, if there are multiple best actions give them all.
            best_actions: list[(int, int)] = [action for action, reward in eval_actions.items() if reward == max(eval_actions.values())]
            # Distribute the probability evenly over the remaining actions.
            action_prob: float = 1 / len(best_actions)
            new_actions: dict[(int, int)] = {}

            for action in best_actions:
                new_actions[action] = action_prob

            # If the actions have updated, mark the policy as stable to continue iterating.
            if old_actions != new_actions:
                policy.movement[s] = new_actions
                policy_stable = False

        # Once the policy is done improving, return the possible rewards.
        if policy_stable:
            return possible_rewards

def plot_dp_heatmap(policy: Policy, possible_rewards) -> None:
    fig, ax = plt.subplots()

    rewards = [[0 for _ in range(policy.grid_size)] for _ in range(policy.grid_size)]
    for coordinates, reward in possible_rewards.items():
        rewards[coordinates[0]][coordinates[1]] = reward

    ax.imshow(rewards)

    ax.set_xticks(range(policy.grid_size))
    ax.set_yticks(range(policy.grid_size))

    arrow_size_multiplier: float = 0.15
    for x in range(policy.grid_size):
        for y in range(policy.grid_size):
            ax.text(x, y, round(possible_rewards[(x, y)], 2), ha="center", va="center", color="w")

            for action in policy.movement[(x, y)]:
                plt.arrow(x, y,
                      action[0] * arrow_size_multiplier,
                      action[1] * arrow_size_multiplier,
                      head_width=0.2,
                      head_length=0.3,
                      fc='k',
                      ec='k')

    ax.set_title("Dynamic Programming Heatmap")
    fig.tight_layout()
    plt.show()