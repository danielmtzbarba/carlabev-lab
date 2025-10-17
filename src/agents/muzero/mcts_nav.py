import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MinMaxStats(object):
    def __init__(self):
        self.max = -float("inf")
        self.min = float("inf")
    
    def update(self, value: float):
        self.max = max(self.max, value)
        self.min = min(self.min, value)
    
    def normalize(self, value: float) -> float:
        if self.max > self.min:
            return (value - self.min) / (self.max - self.min)
        return value

class MCTSNode:
    pb_c_base = 19652
    pb_c_init = 1.25
    def __init__(self, hidden_state, prior):
        self.visit_count = 0
        # Encoded representation of the environment state
        self.hidden_state = hidden_state  
        # Prior probability of taking this action from previous state
        self.prior = prior  
        # Expected value given current hidden_state
        self.value_sum = 0.0
        # Updated mean value from the child nodes
        self.reward = 0.0
        # Expanded action paths from this node
        self.children = {}  

    def ucb_score(self, parent, min_max_stats):
        """
        Standard MuZero UCB equation
        """
        # TODO: Implement the intermediate rewards UCB equation
        discount = 0.99
        pb_c = np.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        pb_c *= np.sqrt(max(parent.visit_count, 0)) / (self.visit_count + 1)

        prior_score = pb_c * self.prior

        if self.visit_count > 0:
            value_score = self.reward + discount * self.value_sum
            value_score =  min_max_stats.normalize(value_score)
        else:
            value_score = 0

        return prior_score + value_score
        
    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    @property
    def isExpanded(self):
        return True if self.hidden_state is not None else False

class MuZeroMCTS:
    def __init__(self, network, action_space_size, num_simulations=50):
        self.muzero = network
        self.dynamics_net = network.dynamics_net
        self.prediction_net = network.policy_head

        self.action_space_size = action_space_size
        self.num_simulations = num_simulations

    def dirichlet_noise(self, policy):
        alpha = 0.3     # Noise concentration
        epsilon = 0.25  # Mixing factor

        dirichlet_noise = np.random.dirichlet([alpha] * self.action_space_size)
        noise_tensor = torch.tensor(
            dirichlet_noise, dtype=policy.dtype, device=policy.device
        )
        return (1 - epsilon) * policy + epsilon * noise_tensor

    def run(self, observation, min_max_stats, isTraining):
        """
        Main function: Runs MCTS starting from an observation
        """
        with torch.no_grad():
            hidden_state, policy_logits, _ = self.muzero.initial_inference(observation)

        policy = F.softmax(policy_logits, dim=1)[0]  # (action_space_size,)

        # Apply Dirichlet noise to encourage exploration (only at root during training)
        if isTraining:
            policy = self.dirichlet_noise(policy)

        # Create root node
        root = MCTSNode(hidden_state, prior=1.0)

        # Expand root
        for action in range(self.action_space_size):
            root.children[action] = MCTSNode(
                hidden_state=None,
                prior=policy[action].item()
            )
        
        # Simulation
        self._simulate(root, min_max_stats)
        
        return root

    def _expand_node(self, node, nn_out):
        next_hidden, policy_logits, value, reward = nn_out
        policy = F.softmax(policy_logits, dim=1)[0]              

        if next_hidden.dim() == 1:
            next_hidden = next_hidden.unsqueeze(0)

        node.hidden_state = next_hidden
        node.reward = reward

        for action in range(self.action_space_size):
            node.children[action] = MCTSNode(
                hidden_state=None,
                prior=policy[action].item()
                )
        return

    def select_action(self, node, min_max_stats):
        """
        Pick the child with highest UCB score and simulate dynamics
        """
        best_score = -float("inf")
        best_action = None

        for action, child in node.children.items():
            ucb = child.ucb_score(node, min_max_stats)
            if ucb > best_score:
                best_score = ucb
                best_action = action

        
        # Add batch dimension
        best_action = (
            torch.tensor(best_action, dtype=torch.long).unsqueeze(0).to("cuda:0"))
        return best_action 

    def _simulate(self, root, min_max_stats):
        """
        Simulate one path through the tree
        """
        for _ in range(self.num_simulations):
            current_node = root 
            path = [current_node]

            # Selection: Traverse the tree
            while current_node.isExpanded:
                # Select next action given actual state
                action = self.select_action(current_node, min_max_stats)
                current_node = current_node.children[action.item()]
                path.append(current_node)

            # Simulate transition
            parent = path[-2]
            with torch.no_grad():
                nn_out = self.muzero.recurrent_inference(
                    parent.hidden_state,
                    torch.tensor([action]).to(device),
                )
            
            # Expand_node
            self._expand_node(current_node, nn_out)

            # Backpropagation
            self._backpropagate(path, nn_out, min_max_stats)


    def _backpropagate(self, path, nn_out, min_max_stats):
        """
        Backpropagate value up the path
        """
        _, _, value, _ = nn_out
        discount = 0.99  # Example discount factor

        for node in reversed(path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.value)
            
            value = node.reward + discount * value
