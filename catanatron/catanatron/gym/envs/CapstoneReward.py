class CapstoneReward:

    def __init__(self, reward_type="simple"):
        REWARD_FUNCS = {
            "simple": self.simple_reward,
            "full": self.full_reward,
        }
        self.reward_func = REWARD_FUNCS.get(reward_type, self.simple_reward)

    def reward(self, game, self_color):
        return self.reward_func(game, self_color)

    def simple_reward(self, game, self_color):
        winning_color = game.winning_color()
        if self_color == winning_color:
            return 1
        elif winning_color is None:
            return 0
        else:
            return -1

    def full_reward(self, game, self_color):
        pass
