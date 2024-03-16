"""
check_multiagent_environments doesn't sample a valid action
    # sample a valid action in case of parametric actions
    if isinstance(reset_obs, dict):
        if config.action_mask_key in reset_obs:
            sampled_action = env.action_space.sample(
                mask=reset_obs[config.action_mask_key]
            )


"""
