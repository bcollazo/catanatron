from catanatron.models.enums import Resource


class ResourceDecks:
    def __init__(self, empty=False):
        starting_amount = 0 if empty else 19
        self.decks = {
            Resource.WOOD: starting_amount,
            Resource.BRICK: starting_amount,
            Resource.SHEEP: starting_amount,
            Resource.WHEAT: starting_amount,
            Resource.ORE: starting_amount,
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

    def __add__(self, other):
        for resource in Resource:
            self.replenish(other.count(resource), resource)
        return self

    def __sub__(self, other):
        for resource in Resource:
            self.draw(other.count(resource), resource)
        return self
