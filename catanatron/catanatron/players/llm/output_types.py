"""
Pydantic models for LLM action outputs.

Supports two modes:
1. Index mode: LLM returns an index into playable_actions (fast, simple)
2. Structured mode: LLM returns typed action with parameters (better for logging/debugging)
"""

from pydantic import BaseModel, Field
from typing import Union, Literal, Optional, Tuple, List


# ============= Index-based Output (Fast Mode) =============


class ActionByIndex(BaseModel):
    """Fast mode: just return the action index from playable_actions list."""

    action_index: int = Field(
        description="Index into the playable_actions list (0-based)"
    )
    reasoning: Optional[str] = Field(
        default=None, description="Brief explanation of why this action was chosen"
    )


# ============= Structured Outputs (Detailed Mode) =============


class RollAction(BaseModel):
    """Roll the dice to start the turn."""

    action_type: Literal["ROLL"] = "ROLL"
    reasoning: str = Field(description="Why rolling now")


class EndTurnAction(BaseModel):
    """End the current turn."""

    action_type: Literal["END_TURN"] = "END_TURN"
    reasoning: str = Field(description="Why ending turn now")


class BuildSettlementAction(BaseModel):
    """Build a settlement at a node."""

    action_type: Literal["BUILD_SETTLEMENT"] = "BUILD_SETTLEMENT"
    node_id: int = Field(description="The node ID where to build the settlement")
    reasoning: str = Field(description="Why building at this location")


class BuildCityAction(BaseModel):
    """Upgrade a settlement to a city."""

    action_type: Literal["BUILD_CITY"] = "BUILD_CITY"
    node_id: int = Field(description="The node ID of the settlement to upgrade")
    reasoning: str = Field(description="Why upgrading this settlement")


class BuildRoadAction(BaseModel):
    """Build a road between two nodes."""

    action_type: Literal["BUILD_ROAD"] = "BUILD_ROAD"
    edge: Tuple[int, int] = Field(description="Edge as (node1_id, node2_id)")
    reasoning: str = Field(description="Why building this road")


class BuyDevelopmentCardAction(BaseModel):
    """Buy a development card from the bank."""

    action_type: Literal["BUY_DEVELOPMENT_CARD"] = "BUY_DEVELOPMENT_CARD"
    reasoning: str = Field(description="Why buying a development card")


class MoveRobberAction(BaseModel):
    """Move the robber to a new tile and optionally steal from a player."""

    action_type: Literal["MOVE_ROBBER"] = "MOVE_ROBBER"
    coordinate: Tuple[int, int, int] = Field(
        description="Tile coordinate (x, y, z) to move robber to"
    )
    victim_color: Optional[str] = Field(
        default=None, description="Color of player to steal from, or None"
    )
    reasoning: str = Field(description="Why moving robber here and stealing from this player")


class DiscardAction(BaseModel):
    """Discard resources when over the limit after a 7 is rolled."""

    action_type: Literal["DISCARD"] = "DISCARD"
    resources: Optional[List[str]] = Field(
        default=None,
        description="List of resources to discard, or None for random discard",
    )
    reasoning: str = Field(description="Why discarding these resources")


# ============= Development Card Actions =============


class PlayKnightCardAction(BaseModel):
    """Play a knight development card."""

    action_type: Literal["PLAY_KNIGHT_CARD"] = "PLAY_KNIGHT_CARD"
    reasoning: str = Field(description="Why playing knight now")


class PlayYearOfPlentyAction(BaseModel):
    """Play Year of Plenty to take 2 resources from the bank."""

    action_type: Literal["PLAY_YEAR_OF_PLENTY"] = "PLAY_YEAR_OF_PLENTY"
    resource1: str = Field(description="First resource to take")
    resource2: str = Field(description="Second resource to take")
    reasoning: str = Field(description="Why choosing these resources")


class PlayMonopolyAction(BaseModel):
    """Play Monopoly to take all of one resource type from other players."""

    action_type: Literal["PLAY_MONOPOLY"] = "PLAY_MONOPOLY"
    resource: str = Field(description="Resource type to monopolize")
    reasoning: str = Field(description="Why choosing this resource")


class PlayRoadBuildingAction(BaseModel):
    """Play Road Building to build 2 free roads."""

    action_type: Literal["PLAY_ROAD_BUILDING"] = "PLAY_ROAD_BUILDING"
    reasoning: str = Field(description="Why playing road building now")


# ============= Trade Actions =============


class MaritimeTradeAction(BaseModel):
    """Trade with the bank at port rates."""

    action_type: Literal["MARITIME_TRADE"] = "MARITIME_TRADE"
    give_resources: List[str] = Field(
        description="Resources to give (2-4 of same type)"
    )
    receive_resource: str = Field(description="Resource to receive")
    reasoning: str = Field(description="Why making this trade")


class OfferTradeAction(BaseModel):
    """Offer a trade to other players."""

    action_type: Literal["OFFER_TRADE"] = "OFFER_TRADE"
    offering: List[int] = Field(
        description="Resources offering as [wood, brick, sheep, wheat, ore] counts"
    )
    asking: List[int] = Field(
        description="Resources asking for as [wood, brick, sheep, wheat, ore] counts"
    )
    reasoning: str = Field(description="Why proposing this trade")


class AcceptTradeAction(BaseModel):
    """Accept a trade offer from another player."""

    action_type: Literal["ACCEPT_TRADE"] = "ACCEPT_TRADE"
    reasoning: str = Field(description="Why accepting this trade")


class RejectTradeAction(BaseModel):
    """Reject a trade offer from another player."""

    action_type: Literal["REJECT_TRADE"] = "REJECT_TRADE"
    reasoning: str = Field(description="Why rejecting this trade")


class ConfirmTradeAction(BaseModel):
    """Confirm a trade with a specific player who accepted."""

    action_type: Literal["CONFIRM_TRADE"] = "CONFIRM_TRADE"
    trading_partner_color: str = Field(
        description="Color of the player to trade with"
    )
    reasoning: str = Field(description="Why trading with this player")


class CancelTradeAction(BaseModel):
    """Cancel the current trade offer."""

    action_type: Literal["CANCEL_TRADE"] = "CANCEL_TRADE"
    reasoning: str = Field(description="Why canceling the trade")


# ============= Union Type for All Actions =============

StructuredActionOutput = Union[
    RollAction,
    EndTurnAction,
    BuildSettlementAction,
    BuildCityAction,
    BuildRoadAction,
    BuyDevelopmentCardAction,
    MoveRobberAction,
    DiscardAction,
    PlayKnightCardAction,
    PlayYearOfPlentyAction,
    PlayMonopolyAction,
    PlayRoadBuildingAction,
    MaritimeTradeAction,
    OfferTradeAction,
    AcceptTradeAction,
    RejectTradeAction,
    ConfirmTradeAction,
    CancelTradeAction,
]

# The main output type - supports both modes
ActionOutput = Union[ActionByIndex, StructuredActionOutput]


# ============= Helper to map action type string to model =============

ACTION_TYPE_TO_MODEL = {
    "ROLL": RollAction,
    "END_TURN": EndTurnAction,
    "BUILD_SETTLEMENT": BuildSettlementAction,
    "BUILD_CITY": BuildCityAction,
    "BUILD_ROAD": BuildRoadAction,
    "BUY_DEVELOPMENT_CARD": BuyDevelopmentCardAction,
    "MOVE_ROBBER": MoveRobberAction,
    "DISCARD": DiscardAction,
    "PLAY_KNIGHT_CARD": PlayKnightCardAction,
    "PLAY_YEAR_OF_PLENTY": PlayYearOfPlentyAction,
    "PLAY_MONOPOLY": PlayMonopolyAction,
    "PLAY_ROAD_BUILDING": PlayRoadBuildingAction,
    "MARITIME_TRADE": MaritimeTradeAction,
    "OFFER_TRADE": OfferTradeAction,
    "ACCEPT_TRADE": AcceptTradeAction,
    "REJECT_TRADE": RejectTradeAction,
    "CONFIRM_TRADE": ConfirmTradeAction,
    "CANCEL_TRADE": CancelTradeAction,
}
