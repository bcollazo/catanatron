use rand::rngs::StdRng;
use rand::{seq::SliceRandom, SeedableRng};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use crate::enums::Resource;
use crate::map_template::{add_coordinates, Coordinate, MapTemplate, TileSlot};

pub type NodeId = u8;
pub type EdgeId = (NodeId, NodeId);

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum NodeRef {
    North,
    NorthEast,
    SouthEast,
    South,
    SouthWest,
    NorthWest,
}

const NODE_REFS: [NodeRef; 6] = [
    NodeRef::North,
    NodeRef::NorthEast,
    NodeRef::SouthEast,
    NodeRef::South,
    NodeRef::SouthWest,
    NodeRef::NorthWest,
];

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum EdgeRef {
    East,
    SouthEast,
    SouthWest,
    West,
    NorthWest,
    NorthEast,
}

const EDGE_REFS: [EdgeRef; 6] = [
    EdgeRef::East,
    EdgeRef::SouthEast,
    EdgeRef::SouthWest,
    EdgeRef::West,
    EdgeRef::NorthWest,
    EdgeRef::NorthEast,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    East,
    SouthEast,
    SouthWest,
    West,
    NorthWest,
    NorthEast,
}

const DIRECTIONS: [Direction; 6] = [
    Direction::East,
    Direction::SouthEast,
    Direction::SouthWest,
    Direction::West,
    Direction::NorthWest,
    Direction::NorthEast,
];

fn get_unit_vector(direction: Direction) -> (i8, i8, i8) {
    match direction {
        Direction::NorthEast => (1, 0, -1),
        Direction::SouthWest => (-1, 0, 1),
        Direction::NorthWest => (0, 1, -1),
        Direction::SouthEast => (0, -1, 1),
        Direction::East => (1, -1, 0),
        Direction::West => (-1, 1, 0),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hexagon {
    pub(crate) nodes: HashMap<NodeRef, NodeId>,
    pub(crate) edges: HashMap<EdgeRef, EdgeId>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LandTile {
    pub(crate) id: u8,
    pub(crate) hexagon: Hexagon,
    pub(crate) resource: Option<Resource>,
    pub(crate) number: Option<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PortTile {
    pub(crate) id: u8,
    pub(crate) hexagon: Hexagon,
    pub(crate) resource: Option<Resource>,
    pub(crate) direction: Direction,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WaterTile {
    pub(crate) hexagon: Hexagon,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Tile {
    Land(LandTile),
    Port(PortTile),
    Water(WaterTile),
}

#[derive(Debug)]
pub struct MapInstance {
    tiles: HashMap<Coordinate, Tile>,
    land_tiles: HashMap<Coordinate, LandTile>,
    port_nodes: HashMap<NodeId, Option<Resource>>,
    adjacent_land_tiles: HashMap<NodeId, Vec<LandTile>>,
    node_production: HashMap<NodeId, HashMap<Resource, f64>>,

    // TODO: Since this doesn't change per map_instance but per template, move
    //  these to MapTemplate.
    // Decided to not use Petgraph for now, since needs seem to be:
    // - Lookup node neighbors of a node
    // - Lookup edges of a node
    // - Lookup list of land edges
    // - Pairwise distances between nodes
    // - Shortest path between nodes
    // - BFS capabilities
    // all which doesn't sound too bad to implement.
    land_nodes: HashSet<NodeId>,

    // TODO: Track valid edges for building roads.
    #[allow(dead_code)]
    land_edges: HashSet<EdgeId>,
    node_neighbors: HashMap<NodeId, Vec<NodeId>>,
    edge_neighbors: HashMap<NodeId, Vec<EdgeId>>,
}

impl MapInstance {
    pub fn get_tiles(&self) -> &HashMap<Coordinate, Tile> {
        &self.tiles
    }

    pub fn get_land_tiles(&self) -> &HashMap<Coordinate, LandTile> {
        &self.land_tiles
    }

    pub fn land_nodes(&self) -> &HashSet<NodeId> {
        &self.land_nodes
    }

    pub fn get_port_nodes(&self) -> &HashMap<NodeId, Option<Resource>> {
        &self.port_nodes
    }

    pub fn get_node_production(&self, node_id: NodeId) -> Option<&HashMap<Resource, f64>> {
        self.node_production.get(&node_id)
    }

    pub fn get_all_node_production(&self) -> &HashMap<NodeId, HashMap<Resource, f64>> {
        &self.node_production
    }

    pub fn get_tile(&self, coordinate: Coordinate) -> Option<&Tile> {
        self.tiles.get(&coordinate)
    }

    pub fn get_land_tile(&self, coordinate: Coordinate) -> Option<&LandTile> {
        self.land_tiles.get(&coordinate)
    }

    pub fn get_neighbor_nodes(&self, node_id: NodeId) -> Vec<NodeId> {
        self.node_neighbors.get(&node_id).unwrap().clone()
    }

    pub fn get_neighbor_edges(&self, node_id: NodeId) -> Vec<EdgeId> {
        self.edge_neighbors.get(&node_id).unwrap().clone()
    }

    pub fn get_adjacent_tiles(&self, node_id: NodeId) -> Option<&Vec<LandTile>> {
        self.adjacent_land_tiles.get(&node_id)
    }

    pub fn get_tiles_by_number(&self, number: u8) -> Vec<&LandTile> {
        self.land_tiles
            .values()
            .filter(|&tile| tile.number == Some(number))
            .collect()
    }
}

impl MapInstance {
    pub fn new(map_template: &MapTemplate, dice_probas: &HashMap<u8, f64>, seed: u64) -> Self {
        let tiles = Self::initialize_tiles(map_template, seed);
        Self::from_tiles(tiles, dice_probas)
    }

    fn initialize_tiles(map_template: &MapTemplate, seed: u64) -> HashMap<Coordinate, Tile> {
        let mut rng = StdRng::seed_from_u64(seed);

        // Shuffle the numbers, tiles, and ports
        let mut shuffled_numbers = map_template.numbers.clone();
        shuffled_numbers.shuffle(&mut rng);
        let mut shuffled_tiles = map_template.tiles.clone();
        shuffled_tiles.shuffle(&mut rng);
        let mut shuffled_ports = map_template.ports.clone();
        shuffled_ports.shuffle(&mut rng);

        // Build the Hexagons by iterating over map_template.topology
        let mut hexagons: HashMap<Coordinate, Hexagon> = HashMap::new();
        let mut tiles: HashMap<Coordinate, Tile> = HashMap::new();
        let mut autoinc = 0;
        let mut tile_autoinc = 0;
        let mut port_autoinc = 0;

        for (&coordinate, &tile_slot) in map_template.topology.iter() {
            let (nodes, edges, new_autoinc) = get_nodes_edges(&hexagons, coordinate, autoinc);
            autoinc = new_autoinc;
            let hexagon = Hexagon { nodes, edges };

            if tile_slot == TileSlot::Land {
                let resource = shuffled_tiles.pop().unwrap();
                if resource.is_none() {
                    let land_tile = LandTile {
                        id: tile_autoinc,
                        hexagon: hexagon.clone(),
                        resource,
                        number: None,
                    };
                    tiles.insert(coordinate, Tile::Land(land_tile));
                } else {
                    let number = shuffled_numbers.pop().unwrap();
                    let land_tile = LandTile {
                        id: tile_autoinc,
                        hexagon: hexagon.clone(),
                        resource,
                        number: Some(number),
                    };
                    tiles.insert(coordinate, Tile::Land(land_tile));
                }
                tile_autoinc += 1;
            } else if tile_slot == TileSlot::Water {
                let water_tile = WaterTile {
                    hexagon: hexagon.clone(),
                };
                tiles.insert(coordinate, Tile::Water(water_tile));
            } else {
                let direction = match tile_slot {
                    TileSlot::NWPort => Direction::NorthWest,
                    TileSlot::NEPort => Direction::NorthEast,
                    TileSlot::EPort => Direction::East,
                    TileSlot::SEPort => Direction::SouthEast,
                    TileSlot::SWPort => Direction::SouthWest,
                    TileSlot::WPort => Direction::West,
                    _ => panic!("Invalid port tile slot"),
                };
                let resource = shuffled_ports.pop().unwrap();
                let port_tile = PortTile {
                    id: port_autoinc,
                    hexagon: hexagon.clone(),
                    resource,
                    direction,
                };
                tiles.insert(coordinate, Tile::Port(port_tile));
                port_autoinc += 1;
            }

            hexagons.insert(coordinate, hexagon);
        }
        tiles
    }

    fn from_tiles(tiles: HashMap<Coordinate, Tile>, dice_probas: &HashMap<u8, f64>) -> Self {
        let mut land_tiles: HashMap<Coordinate, LandTile> = HashMap::new();
        let mut port_nodes: HashMap<NodeId, Option<Resource>> = HashMap::new();
        let mut adjacent_land_tiles: HashMap<NodeId, Vec<LandTile>> = HashMap::new();
        let mut node_production: HashMap<NodeId, HashMap<Resource, f64>> = HashMap::new();

        let mut land_nodes: HashSet<NodeId> = HashSet::new();
        let mut land_edges: HashSet<EdgeId> = HashSet::new();
        let mut node_neighbors: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut edge_neighbors: HashMap<NodeId, Vec<EdgeId>> = HashMap::new();

        for (&coordinate, tile) in tiles.iter() {
            if let Tile::Land(land_tile) = tile {
                land_tiles.insert(coordinate, land_tile.clone());
                let is_desert = land_tile.resource.is_none();
                land_tile.hexagon.nodes.values().for_each(|&node_id| {
                    land_nodes.insert(node_id);
                    adjacent_land_tiles
                        .entry(node_id)
                        .or_default()
                        .push(land_tile.clone());

                    // maybe add this tile's production to the node's production
                    let production = node_production.entry(node_id).or_default();
                    if is_desert {
                        return;
                    }
                    let resource = land_tile.resource.unwrap();
                    let number = land_tile.number.unwrap();
                    let proba = dice_probas.get(&number).unwrap();
                    production.entry(resource).or_insert(0.0);
                    *production.get_mut(&resource).unwrap() += proba;
                });

                land_tile.hexagon.edges.values().for_each(|&edge_id| {
                    land_edges.insert(edge_id);
                    node_neighbors.entry(edge_id.0).or_default().push(edge_id.1);
                    node_neighbors.entry(edge_id.1).or_default().push(edge_id.0);

                    // Only insert edge into edge_neighbors if not already present
                    {
                        let edges_for_node_0 = edge_neighbors.entry(edge_id.0).or_default();
                        if !edges_for_node_0.contains(&edge_id) {
                            edges_for_node_0.push(edge_id);
                        }
                        let edges_for_node_1 = edge_neighbors.entry(edge_id.1).or_default();
                        if !edges_for_node_1.contains(&edge_id) {
                            edges_for_node_1.push(edge_id);
                        }
                    }
                });
            } else if let Tile::Port(port_tile) = tile {
                let (a_noderef, b_noderef) = get_noderefs_from_port_direction(port_tile.direction);
                port_nodes.insert(
                    *port_tile.hexagon.nodes.get(&a_noderef).unwrap(),
                    port_tile.resource,
                );
                port_nodes.insert(
                    *port_tile.hexagon.nodes.get(&b_noderef).unwrap(),
                    port_tile.resource,
                );
            }
        }

        Self {
            tiles,
            land_tiles,
            port_nodes,
            adjacent_land_tiles,
            node_production,

            land_nodes,
            land_edges,
            node_neighbors,
            edge_neighbors,
        }
    }
}

fn get_noderefs_from_port_direction(direction: Direction) -> (NodeRef, NodeRef) {
    match direction {
        Direction::East => (NodeRef::NorthEast, NodeRef::SouthEast),
        Direction::SouthEast => (NodeRef::SouthEast, NodeRef::South),
        Direction::SouthWest => (NodeRef::South, NodeRef::SouthWest),
        Direction::West => (NodeRef::SouthWest, NodeRef::NorthWest),
        Direction::NorthWest => (NodeRef::NorthWest, NodeRef::North),
        Direction::NorthEast => (NodeRef::North, NodeRef::NorthEast),
    }
}

fn get_nodes_edges(
    hexagons: &HashMap<Coordinate, Hexagon>,
    coordinate: Coordinate,
    mut node_autoinc: NodeId,
) -> (HashMap<NodeRef, NodeId>, HashMap<EdgeRef, EdgeId>, NodeId) {
    let mut nodes = HashMap::new();
    let mut edges = HashMap::new();

    // Insert Pre-existing Nodes and Edges
    let neighbor_hexagons: Vec<(Direction, Coordinate)> = DIRECTIONS
        .iter()
        .map(|&direction| {
            let unit_vector = get_unit_vector(direction);
            (direction, add_coordinates(coordinate, unit_vector))
        })
        .collect::<Vec<(Direction, Coordinate)>>();
    for (neighbor_direction, neighbor_coordinate) in neighbor_hexagons {
        if hexagons.contains_key(&neighbor_coordinate) {
            let neighbor_hexagon = hexagons.get(&neighbor_coordinate).unwrap();

            if neighbor_direction == Direction::East {
                nodes.insert(
                    NodeRef::NorthEast,
                    *neighbor_hexagon.nodes.get(&NodeRef::NorthWest).unwrap(),
                );
                nodes.insert(
                    NodeRef::SouthEast,
                    *neighbor_hexagon.nodes.get(&NodeRef::SouthWest).unwrap(),
                );
                edges.insert(
                    EdgeRef::East,
                    *neighbor_hexagon.edges.get(&EdgeRef::West).unwrap(),
                );
            } else if neighbor_direction == Direction::SouthEast {
                nodes.insert(
                    NodeRef::South,
                    *neighbor_hexagon.nodes.get(&NodeRef::NorthWest).unwrap(),
                );
                nodes.insert(
                    NodeRef::SouthEast,
                    *neighbor_hexagon.nodes.get(&NodeRef::North).unwrap(),
                );
                edges.insert(
                    EdgeRef::SouthEast,
                    *neighbor_hexagon.edges.get(&EdgeRef::NorthWest).unwrap(),
                );
            } else if neighbor_direction == Direction::SouthWest {
                nodes.insert(
                    NodeRef::South,
                    *neighbor_hexagon.nodes.get(&NodeRef::NorthEast).unwrap(),
                );
                nodes.insert(
                    NodeRef::SouthWest,
                    *neighbor_hexagon.nodes.get(&NodeRef::North).unwrap(),
                );
                edges.insert(
                    EdgeRef::SouthWest,
                    *neighbor_hexagon.edges.get(&EdgeRef::NorthEast).unwrap(),
                );
            } else if neighbor_direction == Direction::West {
                nodes.insert(
                    NodeRef::NorthWest,
                    *neighbor_hexagon.nodes.get(&NodeRef::NorthEast).unwrap(),
                );
                nodes.insert(
                    NodeRef::SouthWest,
                    *neighbor_hexagon.nodes.get(&NodeRef::SouthEast).unwrap(),
                );
                edges.insert(
                    EdgeRef::West,
                    *neighbor_hexagon.edges.get(&EdgeRef::East).unwrap(),
                );
            } else if neighbor_direction == Direction::NorthWest {
                nodes.insert(
                    NodeRef::North,
                    *neighbor_hexagon.nodes.get(&NodeRef::SouthEast).unwrap(),
                );
                nodes.insert(
                    NodeRef::NorthWest,
                    *neighbor_hexagon.nodes.get(&NodeRef::South).unwrap(),
                );
                edges.insert(
                    EdgeRef::NorthWest,
                    *neighbor_hexagon.edges.get(&EdgeRef::SouthEast).unwrap(),
                );
            } else if neighbor_direction == Direction::NorthEast {
                nodes.insert(
                    NodeRef::North,
                    *neighbor_hexagon.nodes.get(&NodeRef::SouthWest).unwrap(),
                );
                nodes.insert(
                    NodeRef::NorthEast,
                    *neighbor_hexagon.nodes.get(&NodeRef::South).unwrap(),
                );
                edges.insert(
                    EdgeRef::NorthEast,
                    *neighbor_hexagon.edges.get(&EdgeRef::SouthWest).unwrap(),
                );
            } else {
                panic!("Something went wrong");
            }
        }
    }

    // Insert New Nodes and Edges
    for noderef in NODE_REFS {
        if let std::collections::hash_map::Entry::Vacant(e) = nodes.entry(noderef) {
            e.insert(node_autoinc);
            node_autoinc += 1;
        }
    }
    for edgeref in EDGE_REFS {
        edges.entry(edgeref).or_insert_with(|| {
            let (a_noderef, b_noderef) = get_noderefs(edgeref);
            let edge_nodes = (
                *nodes.get(&a_noderef).unwrap(),
                *nodes.get(&b_noderef).unwrap(),
            );
            edge_nodes
        });
    }

    (nodes, edges, node_autoinc)
}

fn get_noderefs(edgeref: EdgeRef) -> (NodeRef, NodeRef) {
    match edgeref {
        EdgeRef::East => (NodeRef::NorthEast, NodeRef::SouthEast),
        EdgeRef::SouthEast => (NodeRef::SouthEast, NodeRef::South),
        EdgeRef::SouthWest => (NodeRef::South, NodeRef::SouthWest),
        EdgeRef::West => (NodeRef::SouthWest, NodeRef::NorthWest),
        EdgeRef::NorthWest => (NodeRef::NorthWest, NodeRef::North),
        EdgeRef::NorthEast => (NodeRef::North, NodeRef::NorthEast),
    }
}

#[cfg(test)]
mod tests {
    use crate::global_state::GlobalState;

    use super::*;

    #[test]
    fn test_get_nodes_edges() {
        let autoinc = 0;
        let (nodes, edges, autoinc) = get_nodes_edges(&HashMap::new(), (0, 0, 0), autoinc);
        assert!(nodes.len() == 6);
        assert!(edges.len() == 6);
        assert_eq!(autoinc, 6);
    }

    #[test]
    fn test_get_nodes_and_edges_for_east_attachment() {
        let mut tiles = HashMap::new();
        let autoinc = 0;
        let (nodes1, edges1, autoinc1) = get_nodes_edges(&tiles, (0, 0, 0), autoinc);

        tiles.insert(
            (0, 0, 0),
            Hexagon {
                nodes: nodes1,
                edges: edges1,
            },
        );
        let (nodes2, edges2, autoinc2) = get_nodes_edges(&tiles, (1, -1, 0), autoinc1);
        assert_eq!(nodes2.len(), 6);
        assert_eq!(nodes2.get(&NodeRef::SouthEast), Some(&8));
        assert_eq!(nodes2.values().max(), Some(&9));
        assert_eq!(edges2.len(), 6);
        assert_eq!(edges2.get(&EdgeRef::East), Some(&(7, 8)));
        assert_eq!(autoinc2, 10);
    }

    // TODO: Test production at a node that has two of the same tiles next to each other.
    // See https://github.com/bcollazo/catanatron/issues/263.

    fn assert_node_value(
        map_instance: &MapInstance,
        coordinates: Coordinate,
        node_ref: NodeRef,
        expected_value: u8,
    ) {
        assert_eq!(
            map_instance
                .land_tiles
                .get(&coordinates)
                .unwrap()
                .hexagon
                .nodes
                .get(&node_ref),
            Some(&expected_value)
        );
    }

    fn assert_land_tile(
        map_instance: &MapInstance,
        coordinates: Coordinate,
        resource: Option<Resource>,
        number: Option<u8>,
    ) {
        let land_tile = map_instance.get_land_tile(coordinates).unwrap();
        assert_eq!(land_tile.resource, resource);
        assert_eq!(land_tile.number, number);
    }

    #[test]
    fn test_map_mini() {
        let global_state = GlobalState::new();
        let map_instance = MapInstance::new(
            &global_state.mini_map_template,
            &global_state.dice_probas,
            0,
        );

        assert_eq!(map_instance.tiles.len(), 19);
        assert_eq!(map_instance.land_tiles.len(), 7);
        assert_eq!(map_instance.land_nodes.len(), 24);
        assert_eq!(map_instance.port_nodes.len(), 0);
        assert_eq!(map_instance.adjacent_land_tiles.len(), 24);
        assert_eq!(map_instance.node_production.len(), 24);

        // Test adjacent_tiles
        let adjacent_tiles = map_instance.adjacent_land_tiles.get(&0).unwrap();
        assert_eq!(adjacent_tiles.len(), 3);
        assert_land_tile(&map_instance, (0, 0, 0), Some(Resource::Ore), Some(9));
        assert_land_tile(&map_instance, (1, 0, -1), Some(Resource::Wheat), Some(4));
        assert_land_tile(&map_instance, (0, 1, -1), Some(Resource::Brick), Some(5));
        // Assert there is a 9 ore in adjacent_tiles
        assert!(adjacent_tiles
            .iter()
            .any(|tile| { tile.resource == Some(Resource::Ore) && tile.number == Some(9) }));
        // Spot-check two more nodes
        assert_eq!(map_instance.adjacent_land_tiles.get(&14).unwrap().len(), 1);
        assert_eq!(map_instance.adjacent_land_tiles.get(&16).unwrap().len(), 2);

        // Test node production
        let node_production = map_instance.node_production.get(&0).unwrap();
        assert_eq!(
            node_production.get(&Resource::Ore),
            Some(&0.1111111111111111)
        );
        assert_eq!(
            node_production.get(&Resource::Wheat),
            Some(&0.08333333333333333)
        );
        assert_eq!(
            node_production.get(&Resource::Brick),
            Some(&0.1111111111111111)
        );

        // Spot-check several node ids
        assert_node_value(&map_instance, (0, 0, 0), NodeRef::North, 0);
        assert_node_value(&map_instance, (0, 0, 0), NodeRef::NorthEast, 1);
        assert_node_value(&map_instance, (1, -1, 0), NodeRef::SouthEast, 8);
        assert_node_value(&map_instance, (1, 0, -1), NodeRef::North, 22);
        assert_node_value(&map_instance, (-1, 0, 1), NodeRef::South, 13);
    }

    #[test]
    fn test_map_instance() {
        let global_state = GlobalState::new();
        let map_instance = MapInstance::new(
            &global_state.base_map_template,
            &global_state.dice_probas,
            0,
        );

        assert_eq!(map_instance.tiles.len(), 37);
        assert_eq!(map_instance.land_tiles.len(), 19);
        assert_eq!(map_instance.land_nodes.len(), 54);
        assert_eq!(map_instance.port_nodes.len(), 18);
        assert_eq!(map_instance.adjacent_land_tiles.len(), 54);

        // Assert tile at 0,0,0 is Land with right resource and number
        assert_eq!(
            map_instance.tiles.get(&(0, 0, 0)),
            Some(&Tile::Land(LandTile {
                id: 0,
                hexagon: Hexagon {
                    nodes: HashMap::from([
                        (NodeRef::North, 0),
                        (NodeRef::NorthEast, 1),
                        (NodeRef::SouthEast, 2),
                        (NodeRef::South, 3),
                        (NodeRef::SouthWest, 4),
                        (NodeRef::NorthWest, 5)
                    ]),
                    edges: HashMap::from([
                        (EdgeRef::East, (1, 2)),
                        (EdgeRef::SouthEast, (2, 3)),
                        (EdgeRef::SouthWest, (3, 4)),
                        (EdgeRef::West, (4, 5)),
                        (EdgeRef::NorthWest, (5, 0)),
                        (EdgeRef::NorthEast, (0, 1))
                    ])
                },
                resource: Some(Resource::Wood),
                number: Some(10)
            }))
        );
        assert_eq!(
            map_instance.land_tiles.get(&(1, -1, 0)),
            Some(&LandTile {
                id: 1,
                hexagon: Hexagon {
                    nodes: HashMap::from([
                        (NodeRef::North, 6),
                        (NodeRef::NorthEast, 7),
                        (NodeRef::SouthEast, 8),
                        (NodeRef::South, 9),
                        (NodeRef::SouthWest, 2),
                        (NodeRef::NorthWest, 1)
                    ]),
                    edges: HashMap::from([
                        (EdgeRef::East, (7, 8)),
                        (EdgeRef::SouthEast, (8, 9)),
                        (EdgeRef::SouthWest, (9, 2)),
                        (EdgeRef::West, (1, 2)),
                        (EdgeRef::NorthWest, (1, 6)),
                        (EdgeRef::NorthEast, (6, 7))
                    ])
                },
                resource: Some(Resource::Ore),
                number: Some(9)
            })
        );

        // Spot-check several node ids
        assert_node_value(&map_instance, (-1, 1, 0), NodeRef::SouthWest, 17);
        assert_node_value(&map_instance, (1, 0, -1), NodeRef::NorthEast, 23);
        assert_node_value(&map_instance, (-1, 2, -1), NodeRef::North, 43);
        assert_node_value(&map_instance, (0, -2, 2), NodeRef::NorthWest, 11);
    }
}

impl Clone for MapInstance {
    fn clone(&self) -> Self {
        Self {
            tiles: self.tiles.clone(),
            land_tiles: self.land_tiles.clone(),
            port_nodes: self.port_nodes.clone(),
            adjacent_land_tiles: self.adjacent_land_tiles.clone(),
            node_production: self.node_production.clone(),
            land_nodes: self.land_nodes.clone(),
            land_edges: self.land_edges.clone(),
            node_neighbors: self.node_neighbors.clone(),
            edge_neighbors: self.edge_neighbors.clone(),
        }
    }
}
