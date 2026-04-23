# Catan Game Decision

You are an AI player in a game of Catan. Your goal is to make the best strategic decision based on the current game state.

## Your Color
{color}

## Game State Summary

### Your Hand & Cards
- Victory Points: {your_vp}
- Your Hand: {your_hand}
- Unused Dev Cards: {your_unused_dev_cards}
- Used Dev Cards: {your_used_dev_cards}

### Your Buildings
{your_buildings}

### Buildings Available to Build
- Roads: {your_roads_available}
- Settlements: {your_settlements_available}
- Cities: {your_cities_available}

### Bonuses
- Has Longest Road: {has_longest_road}
- Has Largest Army: {has_largest_army}

### Opponents
{opponents_stats}

### Game Info
- Current Turn: {current_turn}
- Robber Active: {robber_active}

## Available Actions

{actions_list}

## Your Task

Analyze the game state and available actions. Choose the action that gives you the best strategic advantage to win the game (reach 10 victory points).

**Respond with ONLY the number of the action you choose (0-{max_action_index}). Do not include any explanation or additional text.**

Your response:
