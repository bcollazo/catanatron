//! Design probes, NOT a Catan engine. No third-party dependencies.
use std::{hint::black_box, mem::size_of, time::Instant};

#[derive(Clone)]
struct Fixture {
    player: u8,
    buildings: [u8; 54],
    roads: [u8; 72],
    occupied: u128,
    own: u128,
    expandable: u64,
}
struct Topology {
    edges: [(usize, usize); 72],
    incident: [u128; 54],
    adjacency: Vec<Vec<(usize, usize)>>,
}

fn fixtures() -> (Topology, Vec<Fixture>) {
    let mut lines = include_str!("../fixtures.txt").lines();
    let nums: Vec<usize> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let mut t = Topology {
        edges: [(0, 0); 72],
        incident: [0; 54],
        adjacency: vec![vec![]; 54],
    };
    for e in 0..72 {
        let (a, b) = (nums[2 * e], nums[2 * e + 1]);
        t.edges[e] = (a, b);
        t.incident[a] |= 1 << e;
        t.incident[b] |= 1 << e;
        t.adjacency[a].push((e, b));
        t.adjacency[b].push((e, a));
    }
    let fs = lines
        .map(|line| {
            let n: Vec<u8> = line
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
            let mut f = Fixture {
                player: n[0],
                buildings: n[1..55].try_into().unwrap(),
                roads: n[55..127].try_into().unwrap(),
                occupied: 0,
                own: 0,
                expandable: 0,
            };
            for e in 0..72 {
                if f.roads[e] != 0 {
                    f.occupied |= 1 << e;
                }
                if f.roads[e] == f.player {
                    f.own |= 1 << e;
                }
            }
            for node in 0..54 {
                if f.buildings[node] == f.player
                    || (f.buildings[node] == 0 && t.incident[node] & f.own != 0)
                {
                    f.expandable |= 1 << node;
                }
            }
            f
        })
        .collect();
    (t, fs)
}

fn roads_array(t: &Topology, f: &Fixture) -> u128 {
    let mut result = 0;
    for (e, &(a, b)) in t.edges.iter().enumerate() {
        let expandable = |n: usize| {
            f.buildings[n] == f.player
                || (f.buildings[n] == 0
                    && t.adjacency[n].iter().any(|&(e, _)| f.roads[e] == f.player))
        };
        if f.roads[e] == 0 && (expandable(a) || expandable(b)) {
            result |= 1 << e;
        }
    }
    result
}
fn roads_bits_recompute(t: &Topology, f: &Fixture) -> u128 {
    let mut result = 0;
    for n in 0..54 {
        if f.buildings[n] == f.player || (f.buildings[n] == 0 && t.incident[n] & f.own != 0) {
            result |= t.incident[n];
        }
    }
    result & !f.occupied
}
fn roads_bits_cached(t: &Topology, f: &Fixture) -> u128 {
    let mut nodes = f.expandable;
    let mut result = 0;
    while nodes != 0 {
        let n = nodes.trailing_zeros() as usize;
        nodes &= nodes - 1;
        result |= t.incident[n];
    }
    result & !f.occupied
}

// Both routines compute edge-simple trails, allowing a vertex to be revisited.
// An opponent building terminates an arriving trail but is a legal start/end.
fn longest_vec(t: &Topology, f: &Fixture) -> u8 {
    fn dfs(t: &Topology, f: &Fixture, n: usize, used: Vec<usize>) -> u8 {
        let len = used.len() as u8;
        if len > 0 && f.buildings[n] != 0 && f.buildings[n] != f.player {
            return len;
        }
        let mut best = len;
        for &(e, next) in &t.adjacency[n] {
            if f.roads[e] == f.player && !used.contains(&e) {
                let mut path = used.clone();
                path.push(e);
                best = best.max(dfs(t, f, next, path));
            }
        }
        best
    }
    (0..54).map(|n| dfs(t, f, n, vec![])).max().unwrap()
}
fn longest_bits(t: &Topology, f: &Fixture) -> u8 {
    fn dfs(t: &Topology, f: &Fixture, n: usize, used: u128, len: u8) -> u8 {
        if len > 0 && f.buildings[n] != 0 && f.buildings[n] != f.player {
            return len;
        }
        let mut best = len;
        let mut options = t.incident[n] & f.own & !used;
        while options != 0 {
            let e = options.trailing_zeros() as usize;
            options &= options - 1;
            let (a, b) = t.edges[e];
            best = best.max(dfs(t, f, a ^ b ^ n, used | (1 << e), len + 1));
        }
        best
    }
    (0..54).map(|n| dfs(t, f, n, 0, 0)).max().unwrap()
}

// Full-size payload probes: no boxes, actor carried by the surrounding turn.
#[allow(dead_code)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Action {
    Roll,
    EndTurn,
    Road(u8),
    Settlement(u8),
    City(u8),
    BuyDev,
    Knight,
    Robber { tile: u8, victim: u8 },
    Discard(u8),
    Plenty(u8, u8),
    Monopoly(u8),
    RoadBuilding,
    Maritime { give: u8, receive: u8, rate: u8 },
    Offer([u8; 10]),
    Accept,
    Reject,
    Confirm(u8),
    Cancel,
}
fn score(a: Action) -> u64 {
    match a {
        Action::Road(n) => n as u64,
        Action::Settlement(n) => 100 + n as u64,
        Action::Maritime {
            give,
            receive,
            rate,
        } => 200 + give as u64 + 5 * receive as u64 + 25 * rate as u64,
        Action::Offer(cards) => 400 + cards.iter().map(|&x| x as u64).sum::<u64>(),
        _ => 0,
    }
}
// Only the four workload variants are encoded here. 5-bit counts fit BASE's
// 19 cards/resource. This is not a proposed stable encoding or generic ruleset.
fn pack(a: Action) -> u64 {
    match a {
        Action::Road(n) => (n as u64) << 5,
        Action::Settlement(n) => 1 | ((n as u64) << 5),
        Action::Maritime {
            give,
            receive,
            rate,
        } => 2 | ((give as u64) << 5) | ((receive as u64) << 10) | ((rate as u64) << 15),
        Action::Offer(cards) => cards
            .iter()
            .enumerate()
            .fold(3, |x, (i, &n)| x | ((n as u64) << (5 + 5 * i))),
        _ => panic!("outside probe workload"),
    }
}
fn unpack(x: u64) -> Action {
    match x & 31 {
        0 => Action::Road((x >> 5) as u8),
        1 => Action::Settlement((x >> 5) as u8),
        2 => Action::Maritime {
            give: ((x >> 5) & 31) as u8,
            receive: ((x >> 10) & 31) as u8,
            rate: ((x >> 15) & 31) as u8,
        },
        3 => Action::Offer(std::array::from_fn(|i| ((x >> (5 + 5 * i)) & 31) as u8)),
        _ => unreachable!(),
    }
}

fn bench(name: &str, iterations: usize, mut f: impl FnMut(usize) -> u64) {
    for i in 0..1000 {
        black_box(f(i));
    }
    let mut samples = Vec::new();
    let mut checksum = 0u64;
    for _ in 0..9 {
        let start = Instant::now();
        for i in 0..iterations {
            checksum = checksum.wrapping_add(black_box(f(black_box(i))));
        }
        samples.push(start.elapsed().as_nanos() as f64 / iterations as f64);
    }
    let raw = samples
        .iter()
        .map(|x| format!("{x:.3}"))
        .collect::<Vec<_>>()
        .join(";");
    samples.sort_by(f64::total_cmp);
    println!(
        "{name},{iterations},{:.3},{:.3},{:.3},{checksum},{raw}",
        samples[4], samples[0], samples[8]
    );
}

fn fork_probes<const N: usize>() {
    let states: Vec<[u8; N]> = (0..128)
        .map(|i| std::array::from_fn(|j| (i + j) as u8))
        .collect();
    bench(&format!("copy_mutate_{N}B"), 200_000, |i| {
        let mut s = *black_box(&states[i % states.len()]);
        for j in 0..8 {
            s[j * 7] = s[j * 7].wrapping_add(i as u8);
        }
        black_box(&s)[i % N] as u64
    });
    let mut mutable = states;
    bench(&format!("mutate_undo_{N}B_8bytes"), 200_000, |i| {
        let s = black_box(&mut mutable[i % 128]);
        let old: [u8; 8] = std::array::from_fn(|j| s[j * 7]);
        for j in 0..8 {
            s[j * 7] = s[j * 7].wrapping_add(i as u8);
        }
        let score = black_box(&*s)[i % N] as u64;
        for j in 0..8 {
            s[j * 7] = old[j];
        }
        black_box(s);
        score
    });
}

fn main() {
    let (t, fs) = fixtures();
    for f in &fs {
        assert_eq!(roads_array(&t, f), roads_bits_recompute(&t, f));
        assert_eq!(roads_array(&t, f), roads_bits_cached(&t, f));
        assert_eq!(longest_vec(&t, f), longest_bits(&t, f));
    }
    // Targeted graph regressions beyond random positions: a loop with a tail,
    // the same loop with a blocked junction, and a road ending at an enemy.
    let mut special = Fixture {
        player: 1,
        buildings: [0; 54],
        roads: [0; 72],
        occupied: 0,
        own: 0,
        expandable: 0,
    };
    for &(a, b) in &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (0, 20)] {
        let e = t.edges.iter().position(|&edge| edge == (a, b)).unwrap();
        special.roads[e] = 1;
        special.own |= 1 << e;
    }
    assert_eq!(longest_bits(&t, &special), 7);
    assert_eq!(longest_vec(&t, &special), 7);
    special.buildings[0] = 2;
    assert_eq!(longest_bits(&t, &special), 6);
    assert_eq!(longest_vec(&t, &special), 6);
    special.roads = [0; 72];
    special.roads[0] = 1;
    special.own = 1;
    assert_eq!(longest_bits(&t, &special), 1);
    assert_eq!(longest_vec(&t, &special), 1);
    let actions: Vec<Action> = (0..4096)
        .map(|i| match i % 4 {
            0 => Action::Road((i % 72) as u8),
            1 => Action::Settlement((i % 54) as u8),
            2 => Action::Maritime {
                give: (i % 5) as u8,
                receive: ((i + 1) % 5) as u8,
                rate: 2 + (i % 3) as u8,
            },
            _ => Action::Offer(std::array::from_fn(|j| ((i + j) % 20) as u8)),
        })
        .collect();
    let packed: Vec<u64> = actions.iter().map(|&a| pack(a)).collect();
    for (&a, &p) in actions.iter().zip(&packed) {
        assert_eq!(a, unpack(p));
    }
    eprintln!(
        "Verified {} real-board fixtures; Action={}B packed={}B; parallelism={:?}",
        fs.len(),
        size_of::<Action>(),
        size_of::<u64>(),
        std::thread::available_parallelism()
    );
    println!("name,iterations,median_ns,min_ns,max_ns,checksum,samples_ns");
    bench("loop_control", 300_000, |i| i as u64);
    let fold = |m: u128| (m as u64) ^ ((m >> 64) as u64);
    bench("roads_array_recompute", 100_000, |i| {
        fold(roads_array(black_box(&t), black_box(&fs[i % fs.len()])))
    });
    bench("roads_bits_recompute", 100_000, |i| {
        fold(roads_bits_recompute(
            black_box(&t),
            black_box(&fs[i % fs.len()]),
        ))
    });
    bench("roads_bits_cached_nodes", 100_000, |i| {
        fold(roads_bits_cached(
            black_box(&t),
            black_box(&fs[i % fs.len()]),
        ))
    });
    bench("longest_road_vec_paths", 10_000, |i| {
        longest_vec(black_box(&t), black_box(&fs[i % fs.len()])) as u64
    });
    bench("longest_road_bit_paths", 10_000, |i| {
        longest_bits(black_box(&t), black_box(&fs[i % fs.len()])) as u64
    });
    bench("enum_dispatch", 500_000, |i| {
        score(black_box(actions[i % actions.len()]))
    });
    bench("packed_decode_dispatch", 500_000, |i| {
        score(unpack(black_box(packed[i % packed.len()])))
    });
    let masks: Vec<u128> = fs.iter().map(|f| roads_bits_cached(&t, f)).collect();
    bench("roads_vec_fresh", 100_000, |i| {
        let mut mask = black_box(masks[i % masks.len()]);
        let mut moves = Vec::new();
        while mask != 0 {
            moves.push(Action::Road(mask.trailing_zeros() as u8));
            mask &= mask - 1;
        }
        black_box(&moves).len() as u64
    });
    let mut moves = Vec::with_capacity(72);
    bench("roads_vec_reused", 100_000, |i| {
        moves.clear();
        let mut mask = black_box(masks[i % masks.len()]);
        while mask != 0 {
            moves.push(Action::Road(mask.trailing_zeros() as u8));
            mask &= mask - 1;
        }
        black_box(&moves).len() as u64
    });
    bench("roads_stack_initialized", 100_000, |i| {
        let mut moves = [Action::EndTurn; 72];
        let mut len = 0;
        let mut mask = black_box(masks[i % masks.len()]);
        while mask != 0 {
            moves[len] = Action::Road(mask.trailing_zeros() as u8);
            len += 1;
            mask &= mask - 1;
        }
        black_box(&moves[..len]).len() as u64
    });
    // Precomputed random ranks isolate selection overhead, not RNG throughput.
    bench("roads_select_from_reused_vec", 100_000, |i| {
        moves.clear();
        let mut mask = black_box(masks[i % masks.len()]);
        while mask != 0 {
            moves.push(Action::Road(mask.trailing_zeros() as u8));
            mask &= mask - 1;
        }
        if moves.is_empty() {
            255
        } else {
            score(black_box(moves[i % moves.len()]))
        }
    });
    bench("roads_select_rank_bits", 100_000, |i| {
        let mut mask = black_box(masks[i % masks.len()]);
        let count = mask.count_ones() as usize;
        if count == 0 {
            return 255;
        }
        for _ in 0..i % count {
            mask &= mask - 1;
        }
        mask.trailing_zeros() as u64
    });
    fork_probes::<256>();
    fork_probes::<512>();
    fork_probes::<1024>();
    fork_probes::<4096>();
}
