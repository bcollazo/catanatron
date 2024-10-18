use rand::rngs::StdRng;
use rand::{seq::SliceRandom, SeedableRng};
use std::collections::HashMap;
use std::hash::Hash;

use crate::enums::Resource;
use crate::map_template::{add_coordinates, Coordinate, MapTemplate, TileSlot};

type NodeId = u8;
type EdgeId = (NodeId, NodeId);

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
    // TODO: id?
    pub(crate) nodes: HashMap<NodeRef, NodeId>,
    pub(crate) edges: HashMap<EdgeRef, EdgeId>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LandTile {
    pub(crate) hexagon: Hexagon,
    pub(crate) resource: Option<Resource>,
    pub(crate) number: Option<i8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PortTile {
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
    pub tiles: HashMap<Coordinate, Tile>,
    pub land_tiles: HashMap<Coordinate, LandTile>,
}

impl MapInstance {
    pub fn new(map_template: &MapTemplate, seed: u64) -> Self {
        let tiles = Self::initialize_tiles(map_template, seed);
        Self::from_tiles(tiles)
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
        for (&coordinate, &tile_slot) in map_template.topology.iter() {
            let (nodes, edges, new_autoinc) = get_nodes_edges(&hexagons, coordinate, autoinc);
            autoinc = new_autoinc;
            let hexagon = Hexagon { nodes, edges };

            if tile_slot == TileSlot::Land {
                let resource = shuffled_tiles.pop().unwrap();
                if resource == None {
                    let land_tile = LandTile {
                        hexagon: hexagon.clone(),
                        resource,
                        number: None,
                    };
                    tiles.insert(coordinate, Tile::Land(land_tile));
                } else {
                    let number = shuffled_numbers.pop().unwrap();
                    let land_tile = LandTile {
                        hexagon: hexagon.clone(),
                        resource,
                        number: Some(number),
                    };
                    tiles.insert(coordinate, Tile::Land(land_tile));
                }
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
                    hexagon: hexagon.clone(),
                    resource,
                    direction,
                };
                tiles.insert(coordinate, Tile::Port(port_tile));
            }

            hexagons.insert(coordinate, hexagon);
        }
        tiles
    }

    fn from_tiles(tiles: HashMap<Coordinate, Tile>) -> Self {
        let land_tiles: HashMap<Coordinate, LandTile> = tiles
            .clone()
            .into_iter()
            .filter_map(|(coordinate, tile)| {
                if let Tile::Land(land_tile) = tile {
                    Some((coordinate, land_tile))
                } else {
                    None
                }
            })
            .collect();

        Self { tiles, land_tiles }
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
        if !nodes.contains_key(&noderef) {
            nodes.insert(noderef, node_autoinc);
            node_autoinc += 1;
        }
    }
    for edgeref in EDGE_REFS {
        let (a_noderef, b_noderef) = get_noderefs(edgeref);
        if !edges.contains_key(&edgeref) {
            let edge_nodes = (
                *nodes.get(&a_noderef).unwrap(),
                *nodes.get(&b_noderef).unwrap(),
            );
            edges.insert(edgeref, edge_nodes);
        }
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

    #[test]
    fn test_map_instance() {
        let global_state = GlobalState::new();
        let map_instance = MapInstance::new(&global_state.base_map_template, 0);

        assert_eq!(map_instance.tiles.len(), 37);
        assert_eq!(map_instance.land_tiles.len(), 19);

        // Assert tile at 0,0,0 is Land with right resource and number
        assert_eq!(
            map_instance.tiles.get(&(0, 0, 0)),
            Some(&Tile::Land(LandTile {
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
            map_instance.tiles.get(&(1, -1, 0)),
            Some(&Tile::Land(LandTile {
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
            }))
        );

        // Spot-check several node ids
        assert_node_value(&map_instance, (-1, 1, 0), NodeRef::SouthWest, 17);
        assert_node_value(&map_instance, (1, 0, -1), NodeRef::NorthEast, 23);
        assert_node_value(&map_instance, (-1, 2, -1), NodeRef::North, 43);
        assert_node_value(&map_instance, (0, -2, 2), NodeRef::NorthWest, 11);
    }
}
