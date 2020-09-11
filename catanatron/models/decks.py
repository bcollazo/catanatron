from catanatron.models.enums import Resource


class ResourceDecks:
    def __init__(self):
        self.decks = {
            Resource.WOOD: 19,
            Resource.BRICK: 19,
            Resource.SHEEP: 19,
            Resource.WHEAT: 19,
            Resource.ORE: 19,
        }

    def count(self, resource: Resource):
        return self.decks[resource]

    def can_draw(self, count: int, resource: Resource):
        return self.count(resource) >= count

    def draw(self, count: int, resource: Resource):
        if not self.can_draw(count, resource):
            raise ValueError(f"Not enough resources. Cant draw {count} {resource}")

        self.decks[resource] -= count

    def replenish(self, count: int, resource: Resource):
        self.decks[resource] += count
