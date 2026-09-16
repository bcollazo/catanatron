mod gym_base_2;
mod gym_base_3;
mod gym_base_4;
mod gym_mini_2;
mod gym_mini_3;
mod gym_mini_4;
mod gym_tournament_2;
mod gym_tournament_3;
mod gym_tournament_4;

pub fn gym_keys(map: super::BoardConfig, players: u8) -> &'static [&'static str] {
    match (map, players) {
        (super::BoardConfig::Base, 2) => gym_base_2::KEYS,
        (super::BoardConfig::Base, 3) => gym_base_3::KEYS,
        (super::BoardConfig::Base, 4) => gym_base_4::KEYS,
        (super::BoardConfig::Mini, 2) => gym_mini_2::KEYS,
        (super::BoardConfig::Mini, 3) => gym_mini_3::KEYS,
        (super::BoardConfig::Mini, 4) => gym_mini_4::KEYS,
        (super::BoardConfig::Tournament, 2) => gym_tournament_2::KEYS,
        (super::BoardConfig::Tournament, 3) => gym_tournament_3::KEYS,
        (super::BoardConfig::Tournament, 4) => gym_tournament_4::KEYS,
        _ => &[],
    }
}
